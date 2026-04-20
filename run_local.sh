#!/bin/bash

# TextGuard 本地运行脚本
# 功能：安装uv、配置环境、下载模型、运行main_local.py

echo "=== TextGuard 本地运行脚本 ==="

# 检查并安装uv
if ! command -v uv &> /dev/null; then
    echo "正在安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # 刷新环境变量
    source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null
fi

echo "uv 已安装"

# 配置Hugging Face镜像
echo "配置Hugging Face镜像..."
export HF_ENDPOINT=https://hf-mirror.com

# 同步依赖
echo "同步依赖..."
uv sync

# 创建模型目录
mkdir -p ~/models

# 检查模型是否存在
MODEL_PATH="~/models/Qwen2.5-3B-Instruct"
if [ ! -d "$MODEL_PATH" ]; then
    echo "下载模型Qwen/Qwen2.5-3B-Instruct..."
    uvx hf download Qwen/Qwen2.5-3B-Instruct --local-dir "$MODEL_PATH"
else
    echo "Qwen/Qwen2.5-3B-Instruct模型已存在，跳过下载"
fi

MODEL_PATH="~/models/Qwen3-Embedding-0.6B"
if [ ! -d "$MODEL_PATH" ]; then
    echo "下载模型Qwen/Qwen3-Embedding-0.6B..."
    uvx hf download Qwen/Qwen3-Embedding-0.6B --local-dir "$MODEL_PATH"
else
    echo "Qwen/Qwen3-Embedding-0.6B模型已存在，跳过下载"
fi

# 检查main_local.py是否存在
if [ ! -f "main_local.py" ]; then
    echo "错误：main_local.py 文件不存在"
    echo "请创建main_local.py文件，包含本地模型推理代码"
    exit 1
fi

# 运行main_local.py
echo "运行本地推理..."
uv run main_local.py

echo "=== 运行完成 ==="