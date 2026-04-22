#!/usr/bin/env python3
"""
LoRA微调脚本 - 用于微调qwen2.5/3.5系列模型进行语法纠错
使用sighan2015数据集，训练数据为倒数500条，验证数据为倒数401-500条
"""

import os
import json
import argparse
import sys
sys.path.append('/home/lzh/TextGuard')

from dataclasses import dataclass, field
from typing import Optional, List
from collections import defaultdict
import matplotlib.pyplot as plt

import logging

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# 导入提示词模板
from llm.prompt import GRAMMAR_CHECK_PROMPT

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    # 保存到文件
    filename="./sft/output/train.log"
)


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = field(default="Qwen/Qwen2.5-7B-Instruct", metadata={"help": "模型名称或路径"})
    lora_rank: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    use_bf16: bool = field(default=True, metadata={"help": "是否使用BF16"})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "梯度累积步数"})
    per_device_train_batch_size: int = field(default=4, metadata={"help": "每设备训练批大小"})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "每设备评估批大小"})
    num_train_epochs: int = field(default=3, metadata={"help": "训练轮数"})
    learning_rate: float = field(default=5e-4, metadata={"help": "学习率"})
    warmup_steps: int = field(default=100, metadata={"help": "预热步数"})
    logging_steps: int = field(default=10, metadata={"help": "日志步数"})
    save_steps: int = field(default=100, metadata={"help": "保存步数"})
    output_dir: str = field(default="./sft/output", metadata={"help": "输出目录"})
    plot_loss: bool = field(default=True, metadata={"help": "是否生成loss图表"})
    gpu_ids: str = field(default="0", metadata={"help": "使用的GPU ID，如 '0,1,2,3'"})
    mixed_precision: str = field(default="bf16", metadata={"help": "混合精度: bf16, fp16, none"})


def parse_gpu_ids(gpu_ids_str: str) -> List[int]:
    """解析GPU ID字符串为列表"""
    return [int(x.strip()) for x in gpu_ids_str.split(",")]


def set_gpu_ids(gpu_ids: List[int]):
    """设置可用的GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    logging.info(f"使用GPU: {gpu_ids}")


def get_device_map(num_gpus: int):
    """获取设备映射"""
    device_map = {"": 0}  # 默认将模型放在第一个GPU
    return device_map


class LossPlottingCallback(TrainerCallback):
    """Loss可视化回调"""
    def __init__(self, output_dir, plot_loss=True):
        self.output_dir = output_dir
        self.loss_history = defaultdict(list)
        self.plot_loss = plot_loss

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:

            step = state.global_step
            if "loss" in logs:
                self.loss_history["train_loss"].append((step, logs["loss"]))
            if "eval_loss" in logs:
                self.loss_history["eval_loss"].append((step, logs["eval_loss"]))

    def on_train_end(self, args, state, control, **kwargs):
        if self.plot_loss and self.loss_history["train_loss"]:
            self._plot_loss()

    def _plot_loss(self):
        plt.figure(figsize=(10, 6))

        if self.loss_history["train_loss"]:
            steps, losses = zip(*self.loss_history["train_loss"])
            plt.plot(steps, losses, label="Training Loss")

        if self.loss_history["eval_loss"]:
            steps, losses = zip(*self.loss_history["eval_loss"])
            plt.plot(steps, losses, 'o-', label="Validation Loss")
            best_eval = min(self.loss_history["eval_loss"], key=lambda x: x[1])
            plt.scatter(best_eval[0], best_eval[1], marker='*', s=200, label='Best')

        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss", fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        loss_plot_path = os.path.join(self.output_dir, "loss_plot.png")
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        logging.info(f"\nLoss图表已保存至: {loss_plot_path}")

        # 同时保存loss数据到JSON
        loss_data_path = os.path.join(self.output_dir, "loss_data.json")
        with open(loss_data_path, "w") as f:
            json.dump(dict(self.loss_history), f, indent=2)
        logging.info(f"Loss数据已保存至: {loss_data_path}")


def load_dataset(file_path: str, train_lines: int = 500, val_lines: int = 100) -> tuple:
    """
    加载数据集并按照要求分割
    - 训练数据：倒数train_lines条
    - 验证数据：倒数train_lines到train_lines+val_lines条
    - 测试数据：前面的数据
    """
    logging.info(f"加载数据集: {file_path}")

    # 加载TSV格式数据集
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    logging.info(f"数据集总行数: {total_lines}")

    # 分割数据
    test_end = total_lines - train_lines - val_lines  # 前面的数据用于测试
    train_start = total_lines - train_lines

    test_lines = total_lines - train_lines - val_lines
    train_lines = train_lines
    val_lines = val_lines

    logging.info(f"测试数据: {test_lines}条 (前{test_lines}条)")
    logging.info(f"训练数据: {train_lines}条 (最后{train_lines}条)")
    logging.info(f"验证数据: {val_lines}条 (倒数{val_lines}条)")

    # 分割数据
    test_data = lines[:test_end]  # 前面约107条
    train_data = lines[train_start:total_lines]  # 最后500条 (行207-706)
    val_data = lines[test_end:train_start]  # 倒数100条 (行107-206) - 中间100条

    # 转换为Dataset格式
    from datasets import Dataset
    train_dataset = Dataset.from_list([
        {"source": line.strip().split('\t')[0], "target": line.strip().split('\t')[1]}
        for line in train_data
    ])
    val_dataset = Dataset.from_list([
        {"source": line.strip().split('\t')[0], "target": line.strip().split('\t')[1]}
        for line in val_data
    ])
    test_dataset = Dataset.from_list([
        {"source": line.strip().split('\t')[0], "target": line.strip().split('\t')[1]}
        for line in test_data
    ])

    return train_dataset, val_dataset, test_dataset



def format_prompt(examples: dict) -> dict:
    """
    Qwen 标准对话模板 + 正确标签构建
    """
    conversations = []
    
    for source, target in zip(examples["source"], examples["target"]):
        # 标准 Qwen 模板（官方格式）
        messages = [
            {"role": "system", "content": "你是一个专业的拼写检查器，你的任务是修正句子中的错别字。"},
            {"role": "user", "content": f"请修正这个句子的错别字：{source}"},
            {"role": "assistant", "content": target}  # ✅ 这里是正确答案！
        ]
        
        # 用官方模板拼接（最稳定）
        prompt = "<|im_start|>"
        for message in messages:
            prompt += f"{message['role']}\n{message['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        conversations.append(prompt)

    # 训练时 input 和 labels 都是完整对话（后面tokenize会自动处理）
    return {
        "text": conversations
    }


def batch_tokenize(examples, tokenizer, max_length=256):
    """批量分词（安全版，永不出错）"""
    encodings = tokenizer(
        examples["text"],
        max_length=max_length,
        truncation=True,
        padding="max_length",  # 必须用 max_length，保证长度一致
    )

    input_ids = encodings["input_ids"]
    labels = [ids.copy() for ids in input_ids]

    # 安全查找分隔符，永不出错
    assistant_token = "<|im_start|>assistant\n"
    assistant_ids = tokenizer.encode(assistant_token, add_special_tokens=False)
    match_len = len(assistant_ids)

    for i in range(len(labels)):
        ids = input_ids[i]
        # 安全判断：长度不够直接跳过
        if len(ids) < match_len:
            continue
            
        found = False
        # 只在有效范围内查找
        for j in range(len(ids) - match_len + 1):
            if ids[j:j+match_len] == assistant_ids:
                # 安全赋值
                for k in range(j + match_len):
                    if k < len(labels[i]):
                        labels[i][k] = -100
                found = True
                break
        # 没找到也不报错
        if not found:
            print(f"Warning: 未找到分隔符，样本 {i}")

    return {
        "input_ids": input_ids,
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    }


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="LoRA微调脚本")
    parser.add_argument("--model_name", type=str, default="/home/lzh/models/qwen2.5-7b", help="模型名称或路径")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_bf16", action="store_true", default=True, help="使用BF16")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="每设备训练批大小")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="每设备评估批大小")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=100, help="预热步数")
    parser.add_argument("--logging_steps", type=int, default=1, help="日志步数")
    parser.add_argument("--save_steps", type=int, default=50, help="保存步数")
    parser.add_argument("--output_dir", type=str, default="./sft/output", help="输出目录")
    parser.add_argument("--plot_loss", action="store_true", default=True, help="生成loss图表")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5", help="使用的GPU ID，如 '0,1,2,3'")
    parser.add_argument("--mixed_precision", type=str, default="none", choices=["bf16", "fp16", "none"], help="混合精度量化")

    args = parser.parse_args()

    # 解析GPU ID
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    set_gpu_ids(gpu_ids)

    # 配置
    config = ModelConfig()
    config.model_name = args.model_name
    config.lora_rank = args.lora_rank
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout
    config.use_bf16 = args.use_bf16
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.per_device_train_batch_size = args.per_device_train_batch_size
    config.per_device_eval_batch_size = args.per_device_eval_batch_size
    config.num_train_epochs = args.num_train_epochs
    config.learning_rate = args.learning_rate
    config.warmup_steps = args.warmup_steps
    config.logging_steps = args.logging_steps
    config.save_steps = args.save_steps
    config.output_dir = args.output_dir
    config.plot_loss = args.plot_loss
    config.gpu_ids = args.gpu_ids
    config.mixed_precision = args.mixed_precision

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 1. 加载数据集
    data_path = "/home/lzh/TextGuard/dataset/grammar/sighan2015_test.tsv"
    train_dataset, val_dataset, test_dataset = load_dataset(data_path)

    # 格式化数据
    logging.info("格式化数据...")
    train_dataset = train_dataset.map(format_prompt, batched=True)
    logging.info(f"训练数据样例: {train_dataset[0]}")
    val_dataset = val_dataset.map(format_prompt, batched=True)
    test_dataset = test_dataset.map(format_prompt, batched=True)

    # 2. 加载模型和分词器
    logging.info(f"加载模型: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        local_files_only=True  # 使用本地文件
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    

    # 加载模型
    bnb_config = None
    if config.mixed_precision != "none":
        if config.mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif config.mixed_precision == "fp16":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
    if bnb_config:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )

    # 将模型移动到第一个GPU
    #model = model.to("cuda:0")

    # 3. 设置LoRA (必须在DataParallel之前)
    logging.info("设置LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # # 如果有多个GPU，使用DataParallel包装模型
    # num_gpus = torch.cuda.device_count()
    # if num_gpus > 1:
    #     logging.info(f"使用 {num_gpus} 个GPU进行训练")
    #     model = DataParallel(model)

    # 分词
    logging.info("数据分词中...")

    tokenized_train = train_dataset.map(
        batch_tokenize,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 256},
        batched=True,
        remove_columns=["source", "target", "text"],
    )
    logging.info(f"分词后训练数据列: {tokenized_train.column_names}")

    tokenized_val = val_dataset.map(
        batch_tokenize,
        fn_kwargs={"tokenizer": tokenizer, "max_length": 256},
        batched=True,
        remove_columns=["source", "target", "text"],
    )
    logging.info(f"分词后验证数据列: {tokenized_val.column_names}")

    # 5. 创建数据collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # 6. 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        ddp_find_unused_parameters=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,  # 不移除列，让DataCollator处理
    )

    # 7. 创建回调
    loss_callback = LossPlottingCallback(config.output_dir, config.plot_loss)

    # 8. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=[loss_callback],
    )

    # 8. 开始训练
    logging.info("开始训练...")
    trainer.train()

    # 9. 保存模型
    logging.info("保存模型...")
    best_dir = os.path.join(config.output_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    # 保存当前模型（已经是 best）
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    logging.info(f"Best model 已单独保存至: {best_dir}")

    # 保存训练配置
    config_dict = {
        "model_name": config.model_name,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "num_train_epochs": config.num_train_epochs,
        "learning_rate": config.learning_rate,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    with open(os.path.join(config.output_dir, "training_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    logging.info(f"训练完成！模型保存至: {config.output_dir}")


if __name__ == "__main__":
    main()
