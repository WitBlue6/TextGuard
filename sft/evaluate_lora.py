#!/usr/bin/env python3
"""
评估LoRA微调后的模型
"""

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def format_prompt(source: str) -> str:
    """格式化提示词"""
    return f"<|im_start|>user\n请帮我纠正以下文本的语法错误：\n{source}\n<|im_end|>\n<|im_start|>assistant\n"


def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map=device,
        trust_remote_code=True
    )
    return model, tokenizer


def evaluate_model(model_path: str, test_path: str = "/home/lzh/TextGuard/dataset/grammar/sighan2015_test.tsv", device: str = "auto"):
    """评估模型"""
    print(f"加载模型: {model_path}")

    model, tokenizer = load_model_and_tokenizer(model_path, device)

    # 加载测试数据
    print(f"加载测试数据: {test_path}")
    with open(test_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    test_end = total_lines - 600  # 前面的数据用于测试
    test_data = lines[:test_end]

    results = []
    prompt_template = format_prompt("")

    print(f"开始评估，共 {len(test_data)} 条测试数据...")

    for idx, line in enumerate(test_data):
        if idx % 100 == 0:
            print(f"处理进度: {idx}/{len(test_data)}")

        source, target = line.strip().split('\t')[:2]
        prompt = prompt_template + source

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.1,
                top_p=0.9,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取生成的部分
        if "<|im_end|>assistant" in generated_text:
            generated = generated_text.split("<|im_end|>assistant")[1].strip()
        else:
            generated = generated_text.replace(prompt, "").strip()

        results.append({
            "index": idx,
            "source": source,
            "target": target,
            "generated": generated,
        })

    # 保存结果
    output_file = os.path.join(os.path.dirname(model_path), "evaluation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"评估完成！结果保存至: {output_file}")
    print(f"\n示例结果:")
    print(f"原文: {results[0]['source']}")
    print(f"正确答案: {results[0]['target']}")
    print(f"模型生成: {results[0]['generated']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="微调后的模型路径")
    parser.add_argument("--test_path", type=str, default="/home/lzh/TextGuard/dataset/grammar/sighan2015_test.tsv", help="测试数据路径")
    parser.add_argument("--device", type=str, default="auto", help="设备")

    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_path, args.device)
