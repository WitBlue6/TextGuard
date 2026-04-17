# 本地LLM支持指南

## 概述

TextGuard现在支持两种运行模式：
1. **API模式**：使用远程LLM API（如阿里云通义千问、OpenAI等）
2. **本地模式**：使用本地HuggingFace模型进行推理

## 文件结构

```
TextGuard/
├── llm/                           # LLM相关模块
│   ├── model.py                  # API模式LLM Chain
│   ├── model_local.py            # 本地模式LLM Chain ⭐ 新增
│   ├── entity.py                 # API模式实体管理
│   ├── entity_local.py           # 本地模式实体管理 ⭐ 新增
│   ├── memory.py                 # API模式内存管理
│   └── memory_local.py           # 本地模式内存管理 ⭐ 新增
├── agentic/                      # RAG相关模块
│   ├── router.py                 # API模式路由器
│   ├── router_local.py           # 本地模式路由器 ⭐ 新增
│   ├── indexer.py                # API模式索引器
│   ├── indexer_local.py          # 本地模式索引器 ⭐ 新增
│   ├── retriever.py              # API模式检索器
│   ├── retriever_local.py        # 本地模式检索器 ⭐ 新增
│   ├── evaluator.py              # API模式评估器
│   └── evaluator_local.py        # 本地模式评估器 ⭐ 新增
├── consistency_check.py          # API模式一致性检测
└── consistency_check_local.py    # 本地模式一致性检测 ⭐ 新增
```

## 安装本地模型依赖

```bash
# 安装HuggingFace transformers
pip install transformers torch sentence-transformers

# 对于量化模型（节省内存）
pip install bitsandbytes
```

## 配置说明

### 环境变量

创建 `.env` 文件：

```bash
# API模式配置（可选，本地模式不需要）
OPENAI_API_KEY=sk-xxxx

# 本地模式配置（可选）
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DEVICE=auto  # auto/cuda/cpu/mps
USE_QUANTIZATION=false
USE_RAG=false
```

### 参数说明

运行本地模式时可以使用的参数：

```bash
python consistency_check_local.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \      # LLM模型名称
  --model_path /path/to/local/model \         # 本地模型路径（可选）
  --embedding_model sentence-transformers/all-MiniLM-L6-v2 \  # 嵌入模型
  --device auto \                             # 设备: auto/cuda/cpu/mps
  --llm_quantization \                        # 是否使用量化模型
  --embedding_quantization \                  # 是否量化嵌入模型（实验性）
  --use_rag \                                  # 是否使用RAG增强
  --chroma_db_dir ./chroma_db_local \          # 向量数据库目录
  --docx_data ./dataset/test_long.docx \      # 测试文档
  --log_dir ./logs
```

## 支持的本地模型

### LLM模型

推荐以下HuggingFace模型：

| 模型 | 大小 | 说明 |
|------|------|------|
| Qwen/Qwen2.5-7B-Instruct | 14GB | 推荐使用，中文能力强 |
| Qwen/Qwen2-7B-Instruct | 14GB | Qwen2系列 |
| Baichuan2-7B-Chat | 14GB | 中文模型 |
| Meta-Llama-3-8B-Instruct | 16GB | 英文为主，支持中文 |
| Yi-1.5-9B-Chat | 17GB | 中文能力强 |

**模型下载**：
```bash
# 从HuggingFace下载
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct

# 或使用model_path参数指定路径
python consistency_check_local.py --model_path ./models/Qwen2.5-7B-Instruct
```

### 嵌入模型

推荐以下sentence-transformers模型：

| 模型 | 大小 | 说明 |
|------|------|------|
| sentence-transformers/all-MiniLM-L6-v2 | 120MB | 轻量级，快速 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 470MB | 多语言支持 |
| BAAI/bge-small-zh-v1.5 | 100MB | 中文优化 |

## 使用示例

### 基本使用

```bash
# 使用默认配置（Qwen2.5-7B-Instruct + MiniLM嵌入模型）
python consistency_check_local.py
```

### 指定设备

```bash
# 使用GPU
python consistency_check_local.py --device cuda

# 使用Apple Silicon
python consistency_check_local.py --device mps

# 使用CPU
python consistency_check_local.py --device cpu
```

### 使用RAG增强

```bash
# 启用RAG增强模式
python consistency_check_local.py --use_rag
```

### 使用本地模型路径

```bash
# 如果模型已下载到本地
python consistency_check_local.py \
  --model_path ./models/Qwen2.5-7B-Instruct \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2
```

### 量化模式（节省内存）

```bash
# 使用8bit量化（需要安装bitsandbytes）
python consistency_check_local.py \
  --llm_quantization \
  --device cuda
```

## 模式切换

### API模式

```bash
# 使用API模式（默认）
python consistency_check.py --model_name qwen-plus --base_url https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 本地模式

```bash
# 使用本地模式
python consistency_check_local.py
```

## 功能对比

| 功能 | API模式 | 本地模式 |
|------|---------|----------|
| 语法检查 | ✅ | ✅ |
| 实体提取 | ✅ | ✅ |
| 一致性检查 | ✅ | ✅ |
| RAG增强 | ✅ | ✅ |
| 向量检索 | ✅ | ✅ |
| 人工反馈 | ✅ | ✅ |
| 需要网络 | ✅ | ❌ |
| 需要GPU | ❌（CPU也可） | ✅推荐 |
| 初始设置 | 简单 | 需下载模型 |

## 性能优化建议

### 1. GPU加速

```bash
# 使用CUDA加速
python consistency_check_local.py --device cuda
```

### 2. 模型量化

```bash
# 使用量化模型减少内存占用
python consistency_check_local.py --llm_quantization
```

### 3. 选择合适的嵌入模型

```bash
# 轻量级快速模型
python consistency_check_local.py --embedding_model sentence-transformers/all-MiniLM-L6-v2

# 中文优化模型
python consistency_check_local.py --embedding_model BAAI/bge-small-zh-v1.5
```

### 4. 批量处理

```bash
# 处理大量文档时，适当增加chunk_size
python consistency_check_local.py \
  --chunk_size 2000 \
  --chunk_overlap 300
```

## 常见问题

### Q1: 如何选择LLM模型？

**A**: 推荐使用Qwen2.5-7B-Instruct，它在中文任务上表现良好且性能平衡。如果内存不足，可以使用量化版本。

### Q2: 如何节省内存？

**A**:
1. 使用量化模型：`--llm_quantization`
2. 选择轻量级嵌入模型：`--embedding_model all-MiniLM-L6-v2`
3. 使用CPU模式（虽然较慢）

### Q3: RAG功能如何使用？

**A**:
```bash
python consistency_check_local.py --use_rag
```
RAG会自动创建向量索引并使用检索增强技术。

### Q4: 支持哪些HuggingFace模型？

**A**: 任何支持`AutoModelForCausalLM`和`AutoTokenizer`的模型。推荐使用Instruct模型（如Qwen2.5-7B-Instruct）。

### Q5: 如何查看日志？

**A**: 日志保存在`./logs/consistency_check_local.log`文件中，可以使用以下命令查看：
```bash
tail -f ./logs/consistency_check_local.log
```

## 故障排除

### 1. 模型下载慢

```bash
# 设置HuggingFace镜像（中国用户）
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. CUDA out of memory

```bash
# 使用量化模式
python consistency_check_local.py --llm_quantization --device cuda

# 或使用CPU
python consistency_check_local.py --device cpu
```

### 3. sentence-transformers未安装

```bash
pip install sentence-transformers
```

### 4. 模型推理速度慢

```bash
# 1. 确保使用GPU: --device cuda
# 2. 使用量化模型: --llm_quantization
# 3. 增加batch_size（修改代码中的参数）
```

## 进阶配置

### 自定义设备

```python
# 在代码中指定设备
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 自定义模型参数

```python
# 在llm/model_local.py中修改
pipe = pipeline(
    "text-generation",
    model=model_obj,
    tokenizer=tokenizer,
    device_map=device,
    max_new_tokens=2048,  # 增加最大生成长度
    temperature=0.7,      # 调整温度
    top_p=0.9,
    do_sample=True
)
```

## 性能基准测试

| 配置 | 处理速度 | 内存占用 |
|------|----------|----------|
| Qwen2.5-7B-Instruct (GPU) | ~10 docs/min | ~14GB |
| Qwen2.5-7B-Instruct (CPU) | ~2 docs/min | ~14GB |
| Qwen2.5-7B-Instruct (8bit GPU) | ~15 docs/min | ~8GB |
| MiniLM嵌入 | 快速 | 小 |

## 更新日志

### v1.0.0 (2026-04-17)
- ✨ 初始版本，支持本地LLM和向量模型
- ✨ 支持HuggingFace模型推理
- ✨ 支持sentence-transformers嵌入模型
- ✨ 完整的API模式兼容
- ✨ RAG增强功能支持本地模式
