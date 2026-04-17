# 快速启动指南

## 📦 安装

### 方式1: 使用安装脚本（推荐）

```bash
# 1. 运行安装脚本
./install_local.sh

# 2. 下载模型（可选，脚本中有提示）
# 如果模型已下载，跳过此步

# 3. 运行
./run_local.sh
```

### 方式2: 手动安装

```bash
# 1. 安装依赖
pip install -r requirements_local.txt

# 2. 创建目录
mkdir -p ./models ./chroma_db_local ./logs ./dataset

# 3. 运行
python consistency_check_local.py
```

## 🚀 运行

### 命令行模式

```bash
# 基本运行（使用默认配置）
python consistency_check_local.py

# 使用GPU加速
python consistency_check_local.py --device cuda

# 使用CPU（较慢）
python consistency_check_local.py --device cpu

# 使用RAG增强
python consistency_check_local.py --use_rag

# 使用量化模型（节省内存）
python consistency_check_local.py --llm_quantization --device cuda

# 完整参数示例
python consistency_check_local.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
  --device cuda \
  --use_rag \
  --llm_quantization \
  --chroma_db_dir ./chroma_db_local \
  --docx_data ./dataset/test_long.docx \
  --log_dir ./logs
```

### Web模式

```bash
# 1. 启动Web服务
python main_local.py

# 2. 访问Web界面
# 打开浏览器访问: http://localhost:8001

# 完整参数示例
python main_local.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --rag_mode auto \
  --rag_enabled \
  --log_dir ./logs
```

## 📝 配置

编辑 `local_config.env` 文件：

```bash
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
MODEL_PATH=./models/Qwen2.5-7B-Instruct
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DEVICE=auto
USE_RAG=false
LLM_QUANTIZATION=false
```

## 🔍 查看日志

```bash
# 实时查看日志
tail -f ./logs/consistency_check_local.log

# 实时查看Web日志
tail -f ./logs/web_local.log
```

## 📚 更多信息

- 完整指南: [LOCAL_MODE_GUIDE.md](LOCAL_MODE_GUIDE.md)
- 实现总结: [LOCAL_IMPLEMENTATION_SUMMARY.md](LOCAL_IMPLEMENTATION_SUMMARY.md)
- API文档: [README.md](README.md)

## 🎯 常见使用场景

### 场景1: 快速测试

```bash
python consistency_check_local.py --device cpu
```

### 场景2: 高性能处理

```bash
python consistency_check_local.py --device cuda --llm_quantization
```

### 场景3: 使用RAG增强

```bash
python consistency_check_local.py --use_rag
```

### 场景4: Web界面使用

```bash
python main_local.py --device cuda --rag_enabled
# 访问 http://localhost:8001
```

## ❓ 常见问题

### Q: 如何选择模型？

A: 推荐使用 Qwen2.5-7B-Instruct，它在中文任务上表现良好。

### Q: 内存不足怎么办？

A: 使用量化模式: `--llm_quantization` 或选择更小的模型。

### Q: 如何加速？

A:
1. 使用GPU: `--device cuda`
2. 使用量化: `--llm_quantization`
3. 选择更快的嵌入模型

### Q: RAG如何使用？

A:
```bash
python consistency_check_local.py --use_rag
```
RAG会自动创建向量索引。

## ✅ 完成检查

确认以下步骤完成：

- [ ] 安装了 `sentence-transformers`
- [ ] 下载了LLM模型（可选，可指定路径）
- [ ] 安装了其他依赖
- [ ] 运行了测试命令
- [ ] 查看了日志输出

**祝你使用愉快！** 🎉
