#!/bin/bash

# TextGuard本地模式运行脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  TextGuard 本地模式启动脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 加载环境变量
if [ -f .env ]; then
    echo -e "${YELLOW}加载 .env 配置文件...${NC}"
    source .env
fi

# 设置默认参数
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B-Instruct"}
MODEL_PATH=${MODEL_PATH:-""}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-"sentence-transformers/all-MiniLM-L6-v2"}
DEVICE=${DEVICE:-"auto"}
USE_RAG=${USE_RAG:-"false"}
DOCX_DATA=${DOCX_DATA:-"./dataset/test_long.docx"}
LOG_DIR=${LOG_DIR:-"./logs"}
CHROMA_DB_DIR=${CHROMA_DB_DIR:-"./chroma_db_local"}
LLM_QUANTIZATION=${LLM_QUANTIZATION:-"false"}

# 显示配置
echo -e "${GREEN}配置信息:${NC}"
echo "  LLM模型: $MODEL_NAME"
echo "  嵌入模型: $EMBEDDING_MODEL"
echo "  设备: $DEVICE"
echo "  RAG模式: $USE_RAG"
echo "  数据文件: $DOCX_DATA"
echo "  日志目录: $LOG_DIR"
echo ""

# 检查模型是否已下载
if [ -n "$MODEL_PATH" ] && [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 模型路径不存在: $MODEL_PATH${NC}"
    echo "请先下载模型，或设置正确的模型路径"
    exit 1
fi

# 构建命令
CMD="python consistency_check_local.py"
CMD="$CMD --model_name $MODEL_NAME"
CMD="$CMD --embedding_model $EMBEDDING_MODEL"
CMD="$CMD --device $DEVICE"
CMD="$CMD --docx_data $DOCX_DATA"
CMD="$CMD --log_dir $LOG_DIR"
CMD="$CMD --chroma_db_dir $CHROMA_DB_DIR"

# 添加可选参数
if [ "$USE_RAG" = "true" ]; then
    CMD="$CMD --use_rag"
fi

if [ "$LLM_QUANTIZATION" = "true" ]; then
    CMD="$CMD --llm_quantization"
fi

if [ -n "$MODEL_PATH" ]; then
    CMD="$CMD --model_path $MODEL_PATH"
fi

echo -e "${YELLOW}执行命令:${NC}"
echo "$CMD"
echo ""

# 执行命令
eval $CMD

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  执行成功！${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}  执行失败！${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi
