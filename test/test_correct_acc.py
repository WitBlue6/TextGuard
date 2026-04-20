import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.model import get_grammar_check_chain, get_consistency_correct_chain
import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def parse_args():
    parser = argparse.ArgumentParser(description="Test Grammar Correction Accuracy")
    parser.add_argument("--model_name", type=str, default="gpt-5-mini", help="Model name")
    parser.add_argument("--base_url", type=str, default="https://api.chatanywhere.tech/v1", help="Base URL")
    parser.add_argument("--test_file", type=str, default="./dataset/grammar/sighan2015_test.tsv", help="Test file path")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Output path")
    parser.add_argument("--sample_size", type=int, default=100, help="Sample size for testing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing")
    args = parser.parse_args()
    return args

def logging_config(args):
    # 日志文件路径
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "test_correct_acc.log")

    # 配置 logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 可设置为 DEBUG/INFO 等

    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    # 文件 Handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # 添加 Handler
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

def read_test_file(test_file):
    """
    读取测试文件，支持不同格式
    - train.txt: 格式为 id\terror_type\toriginal\tgold1\tgold2...
    - tsv文件: 格式为 original\tgold
    返回数据列表：[{'id': '1', 'original': '错误句子', 'golds': ['正确句子']}]
    """
    data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')

            # 检查文件格式
            if len(parts) >= 3 and '→' in parts[0]:
                # train.txt格式
                id_str = parts[0]
                original = parts[2]
                golds = parts[3:] if len(parts) > 3 else []
            else:
                # tsv格式
                id_str = str(i+1)
                original = parts[0]
                golds = [parts[1]] if len(parts) > 1 else []

            # 过滤空的gold
            golds = [gold for gold in golds if gold]
            data.append({
                'id': id_str,
                'original': original,
                'golds': golds
            })
    return data

def process_single_item(args, logger, item, grammar_check_chain, judge_chain):
    """
    处理单个样本
    返回: (is_correct, result_dict, item)
    """
    id_str = item['id']
    original = item['original']
    golds = item['golds']

    logger.info(f"[{id_str}] 开始处理")

    # 如果没有gold句子，跳过
    if not golds:
        logger.info(f"[{id_str}] 跳过无gold句子的样本")
        return "SKIPPED", None, item

    result_dict = {}
    corrected = ""
    is_correct = None

    # 语法检查
    retry_count = 0
    while retry_count < 3:
        try:
            user_input = f"【要求】仅修改原始句子中可能存在的错别字/音近字/形近字，保持其他内容不变。\n【原始句子】{original}"
            result = grammar_check_chain.invoke({"new_message": user_input}).content
            if "【" in result:
                result = result.split("【")[-1].strip()
            try:
                result_dict = json.loads(result)
            except json.JSONDecodeError:
                result_dict = {"content": result, "reason": "无法解析结果"}
            corrected = result_dict.get("content", "")
            break
        except Exception as e:
            logger.error(f"[{id_str}] 语法检查失败: {e}，重试次数 {retry_count}")
            retry_count += 1
            time.sleep(3)
            continue

    # 判断是否与gold相近
    for gold in golds:
        retry_count = 0
        while retry_count < 3:
            try:
                judge_result = judge_chain.invoke({
                    "original": original,
                    "corrected": corrected,
                    "gold": gold
                }).content
                if "【" in judge_result:
                    judge_result = judge_result.split("【")[-1].strip()
                if "YES" in judge_result.upper():
                    is_correct = "YES"
                elif "NO" in judge_result.upper():
                    is_correct = "NO"
                else:
                    logger.warning(f"[{id_str}] 未知判断结果: {judge_result}")
                break
            except Exception as e:
                logger.error(f"[{id_str}] 判断失败: {e}")
                retry_count += 1
                time.sleep(3)
                continue
        # 如果找到YES，就不需要继续判断其他gold
        if is_correct == "YES":
            break

    logger.info(f"[{id_str}] 判断结果: {is_correct}")
    return is_correct, result_dict, item

def get_judge_chain(model_name, base_url):
    """
    获取判断链，用于判断修正后的句子是否与gold相近
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个语言判断专家。给定原始句子、修正后的句子和参考正确句子，判断修正后的句子是否与参考正确句子语义相近。\n\n输出格式：\n判断：YES 或 NO\n原因：简短说明"),
        ("human", "原始句子：{original}\n修正后的句子：{corrected}\n参考正确句子：{gold}"),
    ])
    judge_model = ChatOpenAI(
        model_name=model_name,
        temperature=0.1,
        max_tokens=512,  # 增加 tokens 限制
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    judge_chain = judge_prompt | judge_model
    return judge_chain

def test_correct_accuracy(args, logger):
    # 读取测试文件
    logger.info(f"读取测试文件 {args.test_file}")
    data = read_test_file(args.test_file)
    logger.info(f"共读取 {len(data)} 条数据")

    # 限制样本数量
    if args.sample_size > 0:
        len_data = min(len(data), args.sample_size)
        data = data[:len_data]
        logger.info(f"使用 {len_data} 条样本进行测试")

    # 获取语法检查链和判断链
    logger.info("初始化模型链")
    grammar_check_chain = get_grammar_check_chain(args.model_name, args.base_url)
    judge_chain = get_judge_chain(args.model_name, args.base_url)

    # 使用线程安全的计数器
    correct_count = 0
    results = []
    lock = threading.Lock()

    # 分批处理，每批指定数量
    total = len(data)
    batch_count = (total + args.batch_size - 1) // args.batch_size

    logger.info(f"使用 {args.num_workers} 个工作线程进行并发处理")

    for batch_idx in range(batch_count):
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, total)
        batch_data = data[start_idx:end_idx]

        logger.info(f"处理第 {batch_idx + 1}/{batch_count} 批，样本 {start_idx + 1}-{end_idx}")

        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # 提交所有任务
            future_to_item = {
                executor.submit(
                    process_single_item,
                    args,
                    logger,
                    item,
                    grammar_check_chain,
                    judge_chain
                ): item for item in batch_data
            }

            # 收集结果
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    is_correct, result_dict, processed_item = future.result()
                    if is_correct == "SKIPPED":
                        continue

                    # 线程安全地更新计数
                    with lock:
                        if is_correct == "YES":
                            correct_count += 1

                    # 添加到结果
                    with lock:
                        results.append({
                            'id': processed_item['id'],
                            'original': processed_item['original'],
                            'corrected': result_dict.get("content", ""),
                            'reason': result_dict.get("reason", ""),
                            'golds': processed_item['golds'],
                            'is_correct': is_correct
                        })

                except Exception as e:
                    logger.error(f"处理样本 {item['id']} 时出错: {e}")
                    with lock:
                        results.append({
                            'id': item['id'],
                            'original': item['original'],
                            'corrected': "",
                            'reason': "",
                            'golds': item['golds'],
                            'is_correct': "ERROR"
                        })

    # 计算准确率
    accuracy = correct_count / total if total > 0 else 0
    logger.info(f"测试完成，准确率: {accuracy:.4f} ({correct_count}/{total})")

    # 保存结果
    save_dir = os.path.join(args.log_dir, "test_correct_acc")
    os.makedirs(save_dir, exist_ok=True)

    # 根据测试文件名生成结果文件名
    test_filename = os.path.basename(args.test_file)
    result_filename = f"results_{os.path.splitext(test_filename)[0]}.json"
    accuracy_filename = f"accuracy_{os.path.splitext(test_filename)[0]}.txt"

    with open(os.path.join(save_dir, result_filename), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    with open(os.path.join(save_dir, accuracy_filename), "w", encoding="utf-8") as f:
        f.write(f"Test file: {args.test_file}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Correct: {correct_count}\n")
        f.write(f"Total: {total}\n")
        f.write(f"Workers: {args.num_workers}\n")
        f.write(f"Batch size: {args.batch_size}\n")

    return accuracy

if __name__ == "__main__":
    args = parse_args()
    logger = logging_config(args)
    test_correct_accuracy(args, logger)
