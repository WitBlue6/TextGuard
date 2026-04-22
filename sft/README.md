# LoRA微调 - 语法纠错

本项目用于微调qwen2.5/3.5系列模型进行中文语法纠错任务，使用sighan2015数据集。

## 文件说明

- `train_lora.py` - LoRA微调训练脚本
- `evaluate_lora.py` - 模型评估脚本
- `inference_lora.py` - 单条文本推理脚本
- `train.sh` - 训练启动脚本
- `requirements.txt` - 依赖包列表
- `README.md` - 本文档

## 功能特性

1. **每轮验证** - 每轮训练结束后自动运行验证集，得到val loss
2. **可视化图表** - 自动生成training loss和val loss的对比图表
3. **数据保存** - loss数据同时保存为JSON文件和PNG图片
4. **GPU选择** - 支持指定使用的GPU ID（如 0,1,2,3）
5. **自定义提示词** - 使用用户定义的提示词模板（来自llm/prompt.py）

## 数据集说明

数据集路径: `/home/lzh/TextGuard/dataset/grammar/sighan2015_test.tsv`

- 总行数: 707条
- 测试数据: 前面约107条
- 训练数据: 最后500条
- 验证数据: 倒数100条

## 安装依赖

```bash
cd /home/lzh/TextGuard/sft
pip install -r requirements.txt
```

## 训练模型

```bash
# 使用默认配置训练（使用GPU 0）
python train_lora.py

# 指定GPU ID
python train_lora.py --gpu_ids 0

# 使用多个GPU
python train_lora.py --gpu_ids 0,1,2,3

# 指定模型名称
python train_lora.py --model_name Qwen/Qwen2.5-7B-Instruct

# 指定自定义参数
python train_lora.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --lora_rank 16 \
    --lora_alpha 32 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --gpu_ids 0,1

# 快速启动（使用训练脚本）
bash train.sh
```

## 评估模型

```bash
python evaluate_lora.py \
    --model_path ./sft/output \
    --test_path /home/lzh/TextGuard/dataset/grammar/sighan2015_test.tsv
```

## 单条推理

```bash
python inference_lora.py \
    --model_path ./sft/output \
    --text "你朋唷打算去法国玩儿。"
```

输出:
```
原文: 你朋唷打算去法国玩儿。
修正后: {"correct": false, "content": "我今天吃了苹果后去了学校。", "reason": "句子顺序混乱；缺少连接词导致动作逻辑不清晰"}
```

## 输出文件

训练完成后，会在 `output/` 目录下生成：

- `loss_plot.png` - Training loss和Validation loss对比图
- `loss_data.json` - loss数据的JSON文件
- `adapter_config.json` - LoRA配置
- `adapter_model.bin` - 微调后的LoRA权重

## 默认配置

- 模型: `Qwen/Qwen2.5-7B-Instruct`
- LoRA rank: 8
- LoRA alpha: 16
- Dropout: 0.05
- 训练轮数: 3
- 学习率: 5e-4
- 批大小: 4 (每设备)
- 梯度累积: 1
- 保存步数: 100
- GPU: 0

## 提示词说明

代码使用用户定义的提示词模板（来自`llm/prompt.py`中的`GRAMMAR_CHECK_PROMPT`），该提示词要求模型输出严格的JSON格式，包括：
- `correct`: 是否正确
- `content`: 修正后的句子
- `reason`: 错误原因

## 注意事项

1. 训练需要GPU，建议至少8GB显存
2. 使用4bit量化可以减少显存占用
3. 训练完成后会保存模型到 `output` 目录
4. 每轮训练结束后会自动进行验证集评估
5. 训练过程中会自动生成loss图表
