from llm.model import get_grammar_check_chain
from filereader.reader import extract_text_from_pdf, extract_text_from_docx, chunking

import argparse
import os
import logging
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Consistency Check Model")
    parser.add_argument("--model_name", type=str, default="qwen-plus", help="Model name")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="Base URL")
    parser.add_argument("--docx_data", type=str, default="./dataset/test.docx", help="Docs Dataset path")
    #parser.add_argument("--pdf_data", type=str, default="./dataset/test.pdf", help="PDF Dataset path")
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

def check_grammar(args, **kwargs):
    '''
    检查文档中的语法错误
    返回语法错误检测结果列表
    '''
    logger = kwargs.get("logger")

    # chain获取
    grammar_check_chain = get_grammar_check_chain(args.model_name, args.base_url)

    # 文档读取与chunking
    logger.info(f"读取文档 {args.docx_data} ")
    text = extract_text_from_docx(args.docx_data)
    chunks = chunking(text, chunk_size=128)

    # 对每个chunk进行语法检查
    logger.info(f"开始对 {len(chunks)} 个chunk进行语法检查")
    grammar_results = []
    for chunk in chunks:
        result = grammar_check_chain.invoke({"new_message": chunk}).content
        result_dict = json.loads(result)
        result_dict["original_text"] = chunk
        grammar_results.append(result_dict)
        logger.info(f"语法检查结果: {result_dict}")
    
    logger.info(f"语法检查完成，共检查 {len(chunks)} 个chunk")
    # 保存语法检查结果
    save_dir = os.path.join(args.log_dir, os.path.basename(args.docx_data).split(".")[0])
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"保存语法检查结果到 {save_dir}")
    with open(os.path.join(save_dir, "grammar_check_results.json"), "w", encoding="utf-8") as f:
        json.dump(grammar_results, f, ensure_ascii=False, indent=4)
    return grammar_results

if __name__ == "__main__":
    args = parse_args()
    logger = logging_config(args)
    check_grammar(args, logger=logger)
