# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from web import router as chat_router
import argparse
import os
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="WebUI for Text Error Correction")
    parser.add_argument("--model_name", type=str, default="qwen-plus", help="Model name")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="Base URL")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Output path")
    args = parser.parse_args()
    return args

def logging_config(args):
    # 日志文件路径
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "consistency_check.log")

    # 配置 logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 可设置为 DEBUG/INFO 等

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

    logger.info("启动一致性检测服务")
    logger.info(f"模型配置加载完成: model={args.model_name}")

    app = FastAPI(title="文本一致性检测系统", version="0.1.0")

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

    app.include_router(chat_router)
    # 挂载静态文件目录
    app.mount("/", StaticFiles(directory="frontend/static", html=True), name="static")
    logger.info("FastAPI 初始化完成")
    return app

app = create_app()
