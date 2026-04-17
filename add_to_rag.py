#!/usr/bin/env python3
"""
向RAG数据库添加新内容的脚本
支持文档输入或手动字符串输入
"""

import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agentic.indexer import AdvancedIndexer
from filereader.reader import get_text_from_input, chunking

def add_to_rag(text, model_name, base_url, embedding_name, persist_dir):
    """
    向RAG数据库添加新内容
    
    Args:
        text: 要添加的文本内容
        model_name: 模型名称
        base_url: API基础URL
        embedding_name: 嵌入模型名称
        persist_dir: 数据库保存目录
    """
    # 初始化索引器
    indexer = AdvancedIndexer(embedding_name, base_url)
    
    # 对文本进行分块
    chunks = chunking(text, chunk_size=1024)
    print(f"文本分块完成，共 {len(chunks)} 个块")
    
    # 为每个块创建多表示
    all_texts = []
    for i, chunk in enumerate(chunks):
        multi_reps = indexer.create_multi_representation(chunk)
        all_texts.extend(multi_reps)
        print(f"处理第 {i+1} 个块，生成 {len(multi_reps)} 个表示")
    
    print(f"共生成 {len(all_texts)} 个文本表示")
    
    # 创建或更新索引
    vectorstore = indexer.create_raptor_index(all_texts, persist_directory=persist_dir)
    print(f"索引更新完成，保存到: {persist_dir}")
    
    return vectorstore

def main():
    """
    主函数
    """
    # 加载环境变量
    load_dotenv()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="向RAG数据库添加新内容")
    parser.add_argument("--text", type=str, help="要添加的文本内容")
    parser.add_argument("--file", type=str, help="要添加的文档文件路径")
    parser.add_argument("--model_name", type=str, default="qwen3-max", help="模型名称")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API基础URL")
    parser.add_argument("--embedding_name", type=str, default="text-embedding-v4", help="嵌入模型名称")
    parser.add_argument("--persist_dir", type=str, default="./chroma_db", help="数据库保存目录")
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.text and not args.file:
        print("错误: 必须提供 --text 或 --file 参数")
        sys.exit(1)
    
    # 读取文本内容
    text = ""
    if args.text:
        text = args.text
    elif args.file:
        if not os.path.exists(args.file):
            print(f"错误: 文件不存在: {args.file}")
            sys.exit(1)
        # 使用 get_text_from_input 函数读取文件
        text = get_text_from_input("", args.file)
    
    if not text.strip():
        print("错误: 文本内容为空")
        sys.exit(1)
    
    print(f"要添加的文本长度: {len(text)} 字符")
    print("开始添加到RAG数据库...")
    
    # 添加到RAG数据库
    try:
        add_to_rag(
            text=text,
            model_name=args.model_name,
            base_url=args.base_url,
            embedding_name=args.embedding_name,
            persist_dir=args.persist_dir
        )
        print("\n成功: 内容已添加到RAG数据库")
    except Exception as e:
        print(f"\n错误: 添加到RAG数据库失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()