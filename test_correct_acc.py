from llm.model import get_grammar_check_chain, get_consistency_correct_chain
import argparse
import json
import logging
import os
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Test Grammar Correction Accuracy")
    parser.add_argument("--model_name", type=str, default="MiniMax-M2.7", help="Model name")
    parser.add_argument("--base_url", type=str, default="https://api.minimaxi.com/v1", help="Base URL")
    parser.add_argument("--test_file", type=str, default="./dataset/grammar/sighan2015_test.tsv", help="Test file path")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Output path")
    parser.add_argument("--sample_size", type=int, default=100, help="Sample size for testing")
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
        data = data[:args.sample_size]
        logger.info(f"使用 {len(data)} 条样本进行测试")
    
    # 获取语法检查链和判断链
    logger.info("初始化模型链")
    grammar_check_chain = get_grammar_check_chain(args.model_name, args.base_url)
    judge_chain = get_judge_chain(args.model_name, args.base_url)
    
    # 统计结果
    total = len(data)
    correct_count = 0
    results = []
    
    # 对每个样本进行测试
    for i, item in enumerate(data):
        logger.info(f"处理第 {i+1}/{total} 条数据: {item['id']}")
        
        # 原始句子
        original = item['original']
        golds = item['golds']
        
        # 如果没有gold句子，跳过
        if not golds:
            logger.info(f"跳过无gold句子的样本: {item['id']}")
            continue
        
        retry_count = 0
        corrected = ""
        while retry_count < 3:
            # 进行语法检查
            try:
                # 针对数据集场景，只修改错别字
                user_input = f"【要求】仅修改原始句子中可能存在的错别字/音近字/形近字，保持其他内容不变。\n【原始句子】{original}"
                result = grammar_check_chain.invoke({"new_message": user_input}).content
                if "</think>" in result:
                    result = result.split("</think>")[-1].strip()
                result_dict = json.loads(result)
                corrected = result_dict.get("content", "")
                break
            except Exception as e:
                logger.error(f"语法检查失败: {e}，重试次数 {retry_count}, 返回内容: {result}")
                retry_count += 1
                time.sleep(3)  # 等待一段时间后重试
                continue
                
        
        # 判断是否与gold相近
        is_correct = None
        for gold in golds:
            retry_count = 0
            while retry_count < 3:
                # 进行判断
                try:
                    judge_result = judge_chain.invoke({
                        "original": original,
                        "corrected": corrected,
                        "gold": gold
                    }).content
                    if "</think>" in judge_result:
                        judge_result = judge_result.split("</think>")[-1].strip()
                    if "YES" in judge_result.upper():
                        is_correct = "YES"
                    elif "NO" in judge_result.upper():
                        is_correct = "NO"
                    else:
                        logger.warning(f"未知判断结果: {judge_result}")
                    break
                except Exception as e:
                    logger.error(f"判断失败: {e}")
                    retry_count += 1
                    time.sleep(3)  # 等待一段时间后重试
                    continue

        # 统计结果
        if is_correct == "YES":
            correct_count += 1
        
        # 保存结果
        results.append({
            'id': item['id'],
            'original': original,
            'corrected': corrected,
            'reason': result_dict.get("reason", ""),
            'golds': golds,
            'is_correct': is_correct
        })
        
        logger.info(f"判断结果: {is_correct}")
        logger.info(f"原始: {original}")
        logger.info(f"修正: {corrected}")
        logger.info(f"原因: {result_dict.get('reason', '')}")
        logger.info(f"参考: {golds[0]}")
    
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
    
    return accuracy

if __name__ == "__main__":
    args = parse_args()
    logger = logging_config(args)
    test_correct_accuracy(args, logger)