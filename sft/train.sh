#!/bin/bash
# LoRA微调训练脚本

cd /home/lzh/TextGuard/sft

# 激活虚拟环境（根据实际情况修改）
source /home/lzh/TextGuard/.venv/bin/activate

# 安装依赖（如果需要）
pip install -r requirements.txt -q

# 开始训练
python train_lora.py \
    --model_name ~/models/qwen2.5-7b \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_train_epochs 3 \
    --learning_rate 5e-4 \
    --output_dir ./output \
    --gpu_ids 0,1,2,3
