from llm.model import get_feedback_summary_chain
import json
import os
import logging

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

def collect_user_feedback(grammar_results, save_dir, args, logger=None):
    """
    收集用户对语法检查结果的反馈
    
    参数：
    grammar_results: 语法检查结果列表
    save_dir: 保存目录
    args: 命令行参数
    logger: 日志记录器
    
    返回：
    feedback_summary: 反馈总结
    """
    if logger is None:
        logger = logging.getLogger()
    
    logger.info("开始收集用户反馈")
    
    # 显示语法检查结果
    logger.info("\n=== 语法检查结果 ===")
    for i, result in enumerate(grammar_results):
        logger.info(f"\nChunk {i+1}:")
        logger.info(f"原始内容: {result.get('original_text', 'N/A')}")
        logger.info(f"修改后: {result.get('content', 'N/A')}")
        logger.info(f"是否正确: {result.get('correct', 'N/A')}")
        logger.info(f"错误原因: {result.get('reason', 'N/A')}")
    
    # 收集用户反馈
    user_feedback = input("\n请输入您的修改意见或反馈（按Enter键确认）：")
    
    if not user_feedback.strip():
        logger.info("未收到用户反馈，跳过反馈总结")
        return None
    
    # 调用LLM总结反馈
    logger.info("调用LLM总结用户反馈")
    feedback_chain = get_feedback_summary_chain(args.model_name, args.base_url)
    
    # 准备输入
    input_data = {
        "grammar_results": str(grammar_results),
        "user_feedback": user_feedback
    }
    
    # 获取总结结果
    summary_result = feedback_chain.invoke(input_data).content
    
    # 保存反馈和总结
    feedback_data = {
        "user_feedback": user_feedback,
        "feedback_summary": summary_result,
        "grammar_results": grammar_results
    }
    
    feedback_file = os.path.join(save_dir, "user_feedback.json")
    with open(feedback_file, "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, ensure_ascii=False, indent=4)
    
    logger.info(f"用户反馈已保存到 {feedback_file}")
    logger.info(f"反馈总结: {summary_result}")
    
    return summary_result

if __name__ == "__main__":
    import argparse
    
    def parse_args():
        parser = argparse.ArgumentParser(description="Feedback Collection")
        parser.add_argument("--model_name", type=str, default="qwen-plus", help="Model name")
        parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="Base URL")
        parser.add_argument("--log_dir", type=str, default="./logs", help="Output path")
        parser.add_argument("--docx_data", type=str, default="./dataset/test.docx", help="Docs Dataset path")
        args = parser.parse_args()
        return args
    
    args = parse_args()
    save_dir = args.log_dir
    
    # 模拟语法检查结果
    mock_results = [
        {
            "correct": False,
            "content": "在班会课上却展示了一张自己获得全市演讲比赛一等奖的奖状，并说自己从小学就是辩论队队长，最喜欢在大庭广众下发言。但当同学们问他要不要加入学校的演讲队时，他却说自己一上台就会紧张得说不出话，从来不敢参加这类活动。",
            "reason": "介宾结构误用：'在班会课时'应为'在班会课上'；成分赘余：'下发言'应为'下发言'以避免重复",
            "original_text": "在班会课时却⼜展示了⼀张⾃⼰获得全市演讲⽐赛⼀等奖的奖状，并说⾃⼰从⼩学就是辩论队队⻓，最喜欢在⼤庭⼴众下发⾔。但当同学们问他要不要加⼊学校的演讲队时，他却说⾃⼰⼀上台就会紧张得说不出话，从来不敢参加这类活动。"
        }
    ]
    # 配置日志记录
    logger = logging_config(args)
    
    collect_user_feedback(mock_results, save_dir, args, logger)
