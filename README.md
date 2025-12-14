# 文本一致性检测与中文语法纠错系统

一个基于大语言模型和LangChain开发的中文文本处理系统，提供实体一致性检测和中文语法纠错功能。

## 功能特性

- **中文语法纠错**：自动检测并纠正文本中的语法错误
- **文本实体提取**：自动识别文本中的关键实体
- **实体一致性检查**：检测实体在文本中的描述是否一致
- **多格式支持**：支持直接输入文本或上传docx/pdf文件
- **实时通信**：基于WebSocket的实时检测进度反馈
- **友好界面**：现代化的Web界面，操作简单直观

## 技术栈

- **后端框架**：FastAPI
- **前端技术**：HTML5, JavaScript
- **大语言模型**：阿里云通义千问 (qwen-plus)
- **LLM应用框架**：LangChain
- **文件处理**：支持docx和pdf格式
- **包管理**：uv（Python包管理工具）
- **部署方式**：本地服务器

## 安装说明

### 环境要求

- Python 3.10+
- uv 包管理工具

### 安装步骤

1. 克隆项目

```bash
git clone <repository-url>
cd CGEC
```

2. 安装依赖（使用uv）

```bash
uv sync
```

3. 配置环境变量

在项目根目录创建`.env`文件，配置模型相关参数：

```dotenv
# 模型配置
OPENAI_API_KEY=sk-xxxx
```

## 使用方法

### 启动服务

```bash
# 使用uv运行
cd CGEC
uv run run.py
```

服务将在`http://localhost:8000`启动

### Web界面使用

1. 打开浏览器访问`http://localhost:8000`
2. 在文本框中输入要检测的文本，或点击"选择文件"上传docx/pdf文件
3. 点击"发送"按钮开始检测
4. 在日志区域查看实时检测进度和结果

### API使用

#### HTTP API

```bash
POST /chat
```

参数：
- `message`: 文本内容（可选）
- `history`: 历史对话记录（可选）
- `file`: 上传的文件（可选，docx/pdf格式）

#### WebSocket API

```bash
ws://localhost:8000/ws/chat
```

发送消息格式：
```json
{
  "message": "要检测的文本",
  "file": {
    "filename": "文件名",
    "content": "base64编码的文件内容"
  }
}
```

## 项目结构

``` plaintext
├── README.md
├── consistency_check.py   # 语义一致性检测
├── dataset                # 数据集
├── filereader             # PDF/DOCX文件读取模块
│   ├── __init__.py
│   └── reader.py
├── frontend               # Web前端界面
│   └── static
│       ├── chat.js
│       └── index.html
├── llm                    # Langchain相关模块
│   ├── __init__.py
│   ├── entity.py          # 实体抽取模块
│   ├── memory.py          # 记忆管理模块
│   ├── model.py           # Chain定义
│   └── prompt.py          # SP模版定义
├── logs
├── main.py                # 主应用入口
├── pyproject.toml
├── run.py                 # 启动脚本
├── test                   # 测试脚本
│   ├── test_model.py
│   └── test_reader.py
├── uv.lock
└── web.py                 # FastAPI应用入口
```

## 配置说明

可以通过命令行参数或环境变量配置系统：

| 参数名 | 类型 | 默认值 | 说明 |
|-------|------|-------|------|
| --model_name | str | qwen-plus | 使用的模型名称 |
| --base_url | str | https://dashscope.aliyuncs.com/compatible-mode/v1 | 模型API地址 |
| --log_dir | str | ./logs | 日志文件目录 |

## 工作原理

1. **文本处理**：将输入文本分块处理
2. **语法纠错**：使用LangChain构建的语法纠错Chain检测并纠正文本中的语法错误
3. **实体提取**：从每个文本块中提取实体信息
4. **记忆管理**：利用LangChain的记忆机制维护实体的上下文信息
5. **一致性检查**：使用LangChain Chain对每个实体进行跨文本的一致性检查
6. **结果输出**：返回检查结果和详细日志

## 开发说明

### 前端开发

前端文件位于`frontend/static/`目录下，可以直接修改HTML和JS文件进行定制。

### 后端开发

- API路由定义在`web.py`
- 核心逻辑位于`consistency_check.py`
- 基于LangChain的模型调用相关代码在`llm/`目录下
  - `model.py`: 定义了各种LangChain Chain（语法纠错、实体提取、一致性检查等）
  - `entity.py`: 实体管理相关功能
  - `memory.py`: 上下文记忆管理
  - `prompt.py`: 提示模板定义

## 依赖管理

项目使用uv进行依赖管理，配置文件为`pyproject.toml`，依赖锁定文件为`uv.lock`。

## 日志

系统日志保存在`logs/consistency_check.log`文件中，可以通过配置调整日志级别和格式。

## 许可证

MIT License

## 联系方式

如有问题或建议，请联系项目维护者。