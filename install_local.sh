#!/bin/bash

# TextGuard 本地模式安装脚本

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  TextGuard 本地模式安装脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查Python版本
echo -e "${YELLOW}[1/4] 检查Python版本...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $PYTHON_VERSION"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${RED}错误: 需要Python 3.8或更高版本${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python版本符合要求${NC}"
echo ""

# 安装依赖
echo -e "${YELLOW}[2/4] 安装本地模式依赖...${NC}"
echo -e "${BLUE}安装HuggingFace相关包...${NC}"
pip3 install --upgrade pip setuptools wheel

echo -e "${BLUE}安装transformers和torch...${NC}"
pip3 install transformers torch --index-url https://download.pytorch.org/whl/cpu

echo -e "${BLUE}安装sentence-transformers...${NC}"
pip3 install sentence-transformers

echo -e "${BLUE}安装LangChain相关包...${NC}"
pip3 install langchain langchain-core langchain-community langchain-huggingface

echo -e "${BLUE}安装ChromaDB...${NC}"
pip3 install chromadb

echo -e "${BLUE}安装其他依赖...${NC}"
pip3 install pydantic python-dotenv jieba numpy scikit-learn tqdm

echo -e "${GREEN}✓ 依赖安装完成${NC}"
echo ""

# 创建必要的目录
echo -e "${YELLOW}[3/4] 创建目录结构...${NC}"
mkdir -p ./models
mkdir -p ./chroma_db_local
mkdir -p ./logs
mkdir -p ./dataset
echo -e "${GREEN}✓ 目录创建完成${NC}"
echo ""

# 模型下载
echo -e "${YELLOW}[4/4] 下载推荐模型...${NC}"
echo -e "${BLUE}下载嵌入模型 (all-MiniLM-L6-v2)...${NC}"
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('✓ 嵌入模型下载完成')
"

echo -e "${BLUE}下载LLM模型 (Qwen2.5-7B-Instruct)...${NC}"
echo -e "${YELLOW}注意: 这是一个较大的模型 (~14GB)，下载需要一些时间${NC}"
echo "推荐使用命令行下载，或跳过此步骤手动下载:"
echo ""
echo -e "${BLUE}  # 从HuggingFace下载${NC}"
echo -e "${BLUE}  huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct${NC}"
echo ""
echo -e "${BLUE}  # 或使用model_path参数指定已下载的模型路径${NC}"
echo -e "${BLUE}  python consistency_check_local.py --model_path ./models/Qwen2.5-7B-Instruct${NC}"
echo ""

# 完成提示
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  安装完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}后续步骤:${NC}"
echo "1. 下载LLM模型到 ./models/Qwen2.5-7B-Instruct 或指定model_path"
echo "2. 下载嵌入模型（已完成）"
echo "3. 运行脚本: ./run_local.sh"
echo "4. 或直接运行: python consistency_check_local.py"
echo ""
echo -e "${YELLOW}快速开始:${NC}"
echo -e "${YELLOW}  python consistency_check_local.py --device cpu${NC}"
echo ""
echo -e "${YELLOW}使用GPU加速:${NC}"
echo -e "${YELLOW}  python consistency_check_local.py --device cuda --llm_quantization${NC}"
echo ""
echo -e "${YELLOW}使用RAG增强:${NC}"
echo -e "${YELLOW}  python consistency_check_local.py --use_rag${NC}"
echo ""
