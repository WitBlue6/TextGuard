# run.py
import uvicorn
import signal
import sys
import asyncio
from main import app

# 处理信号，确保程序能被正确终止
def handle_signal(signum, frame):
    print("\n收到终止信号，正在关闭服务器...")
    # 关闭uvicorn服务器
    sys.exit(0)

if __name__ == "__main__":
    # 注册信号处理
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # 启动服务器，建议在生产环境关闭reload
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)