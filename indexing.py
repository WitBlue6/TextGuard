from agentic.indexer import AdvancedIndexer

import argparse
import logging
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

def parse_args():
    parser = argparse.ArgumentParser(description="Consistency Check Model")
    parser.add_argument("--model_name", type=str, default="qwen-plus", help="Model name")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="Base URL")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Output path")
    parser.add_argument("--chroma_db_dir", type=str, default="./chroma_db", help="Chroma DB path")
    parser.add_argument("--txt_data", type=str, default="./dataset/rag.txt", help="Dataset path")
    args = parser.parse_args()
    return args

def logging_config(args):
    # 日志文件路径
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "indexing.log")

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

def extract_text_from_txt(file_path):
    """
    从TXT文件提取文本
    :param file_path: TXT文件路径
    :return: 提取的文本
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"TXT 文本提取失败: {e}")
        return ""
 
 
def load_text_from_file(file_path):
    """
    从文件加载文本
    :param file_path: 文件路径
    :return: 提取的文本
    """
    logger.info(f"读取文件: {file_path}")
 
    if file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
 
 
# 文本分块函数
def chunking(text, chunk_size=1000, chunk_overlap=200):
    """
    将文本切分成块
    :param text: 输入文本
    :param chunk_size: 块大小
    :param chunk_overlap: 块重叠大小
    :return: 文本块列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", "", " "]
    )
    return text_splitter.split_text(text)
 

def create_raptor_index(args, chunks, **kwargs):
    """
    创建Raptor索引
    :param args: 命令行参数
    :param chunks: 输入的文本块列表
    :param logger: 日志记录器
    :return: Raptor索引
    """
    logger = kwargs.get("logger")
    advanced_indexer = AdvancedIndexer(args.model_name, args.base_url)
    # 创建多表示索引并构建Raptor索引
    logger.info("创建多表示索引...")
    all_texts = []
    for chunk in chunks:
        multi_reps = advanced_indexer.create_multi_representation(chunk)
        all_texts.extend(multi_reps)
    
    logger.info("创建Raptor索引...")
    raptor_index = advanced_indexer.create_raptor_index(all_texts, persist_directory=args.chroma_db_dir)
    
    return raptor_index


if __name__ == "__main__":
    args = parse_args()
    logger = logging_config(args)

    logger.info(f"开始创建索引，模型: {args.model_name}, 数据集: {args.txt_data}")

    text = load_text_from_file(args.txt_data)

    chunks = chunking(text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    # 显示前几个块的信息
    for i, chunk in enumerate(chunks[:3]):
        logger.info(f"Chunk {i+1} 内容预览: {chunk[:100]}... (长度: {len(chunk)})")

    # 创建索引
    raptor_index = create_raptor_index(args, chunks, logger=logger)

    if raptor_index:
        logger.info("Raptor索引创建成功")
    else:
        logger.error("Raptor索引创建失败")