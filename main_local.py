# main_local.py
# 本地模式主入口
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from web.chat_local import router as chat_local_router
import argparse
import os
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="WebUI for TextGuard (Local Mode)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="LLM模型名称")
    parser.add_argument("--model_path", type=str, default="/home/lzh/models/qwen2.5-7b", help="本地模型路径")
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="嵌入模型名称")
    parser.add_argument("--embedding_model_path", type=str, default="/home/lzh/models/qwen3-embedding-0.6b", help="本地嵌入模型路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu/mps)")
    parser.add_argument("--llm_quantization", action="store_true", help="使用量化模型")
    parser.add_argument("--rag_mode", type=str, default="none", choices=["none", "mandatory", "auto"], help="RAG模式")
    parser.add_argument("--rag_enabled", action="store_true", help="启用RAG")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Output path")
    parser.add_argument("--chroma_db_dir", type=str, default="./chroma_db_local", help="ChromaDB Save Path")
    args = parser.parse_args()
    return args

def logging_config(args):
    # 日志文件路径
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "web_local.log")

    # 配置 logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    # 文件 Handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # 添加 Handler
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def create_app() -> FastAPI:
    args = parse_args()
    logger = logging_config(args)

    logger.info("启动本地一致性检测Web服务")
    logger.info(f"模型配置: {args.model_name}")
    logger.info(f"设备: {args.device}")
    logger.info(f"RAG模式: {args.rag_mode}")
    logger.info(f"RAG启用: {args.rag_enabled}")

    app = FastAPI(title="TextGuard本地模式", version="1.0.0-local")

    # 全局状态挂载
    app.state.logger = logger
    app.state.args = args

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat_local_router)
    # 挂载静态文件目录
    app.mount("/", StaticFiles(directory="frontend/static", html=True), name="static")
    logger.info("FastAPI初始化完成")
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_local:app", host="127.0.0.1", port=8001, reload=False)
