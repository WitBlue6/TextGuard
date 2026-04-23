#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
性能基准测试脚本
用于测试不同文本长度下语法纠错和一致性检测的响应时间
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import statistics
import json
from typing import List, Tuple, Dict
from dataclasses import dataclass
import argparse

from llm.model import get_grammar_check_chain, get_entity_consistency_check_chain, get_entity_extract_chain, get_memory_summary_chain
from llm.entity import EntityStore, extract_entities, summarize_entity_memory
from filereader.reader import chunking, extract_text_from_docx
import logging
import matplotlib.pyplot as plt


@dataclass
class PerformanceResult:
    """性能测试结果"""
    text_length: int
    num_words: int
    total_time: float
    avg_time: float
    std_dev: float
    min_time: float
    max_time: float
    runs: int

def measure_grammar_check_performance(text: str, runs: int = 3, model_name: str = None, base_url: str = None) -> PerformanceResult:
    """
    测量语法纠错性能
    
    Args:
        text: 输入文本
        runs: 运行次数
    
    Returns:
        性能测试结果
    """
    # 初始化链
    grammar_check_chain = get_grammar_check_chain(model_name, base_url)
    
    times = []
    text_length = len(text)
    num_words = len(text.split())
    
    for i in range(runs):
        start_time = time.time()
        
        # 将文本分块处理（模拟实际使用场景）
        chunks = chunking(text, chunk_size=1024)
        
        for chunk in chunks:
            result = grammar_check_chain.invoke({"new_message": chunk}).content
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    # 计算统计信息
    total_time = sum(times)
    avg_time = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    
    return PerformanceResult(
        text_length=text_length,
        num_words=num_words,
        total_time=total_time,
        avg_time=avg_time,
        std_dev=std_dev,
        min_time=min_time,
        max_time=max_time,
        runs=runs
    )

def measure_consistency_check_performance(text: str, runs: int = 3, model_name: str = None, base_url: str = None) -> PerformanceResult:
    """
    测量一致性检测性能
    
    Args:
        text: 输入文本
        runs: 运行次数
    
    Returns:
        性能测试结果
    """
    # 初始化链
    entity_extract_chain = get_entity_extract_chain(model_name, base_url)
    entity_consistency_check_chain = get_entity_consistency_check_chain(model_name, base_url)
    memory_summary_chain = get_memory_summary_chain(model_name, base_url)
    
    times = []
    text_length = len(text)
    num_words = len(text.split())
    
    for i in range(runs):
        start_time = time.time()
        
        # 将文本分块处理
        chunks = chunking(text)
        
        # 提取实体
        ent_store = EntityStore()
        previous_memory = ""
        
        for j, chunk in enumerate(chunks):
            chunk_input = f"前文要点总结:{previous_memory}\n当前输入文本:{chunk}" if previous_memory else chunk
            
            # 提取本chunk的实体
            ents = extract_entities(entity_extract_chain, chunk_input)
            for ent in ents:
                ent_store.add_entity(ent)
            
            if j < len(chunks) - 1:
                # 更新memory，用于下一chunk
                previous_memory = summarize_entity_memory(memory_summary_chain, chunk_input)
        
        # 检查实体一致性
        for ent in ent_store.all_entities():
            entity_description = f"实体名称: {ent.name}\n实体类型: {ent.type}\n实体属性: {ent.attributes}\n实体事件: {ent.events}\n实体关系: {ent.relations}"
            entity_consistency_check_chain.invoke({"new_message": entity_description})
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    # 计算统计信息
    total_time = sum(times)
    avg_time = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    
    return PerformanceResult(
        text_length=text_length,
        num_words=num_words,
        total_time=total_time,
        avg_time=avg_time,
        std_dev=std_dev,
        min_time=min_time,
        max_time=max_time,
        runs=runs
    )

def run_performance_tests(text_lengths: List[int], runs_per_test: int = 3, model_name: str = None, base_url: str = None) -> Dict[str, List[PerformanceResult]]:
    """
    运行性能测试
    
    Args:
        text_lengths: 要测试的文本长度列表（词数）
        runs_per_test: 每个测试运行的次数
    
    Returns:
        包含语法纠错和一致性检测性能结果的字典
    """
    results = {
        "grammar_check": [],
        "consistency_check": []
    }
    
    for length in text_lengths:
        print(f"正在测试长度为 {length} 词的文本...")
        
        # 文本path
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        text_path = f"{base_dir}/dataset/performance/{length}.docx"
        test_text = extract_text_from_docx(text_path)
        
        # 测试语法纠错
        print(f"  语法纠错测试...")
        grammar_result = measure_grammar_check_performance(test_text, runs_per_test, model_name, base_url)
        results["grammar_check"].append(grammar_result)
        
        # 测试一致性检测
        print(f"  一致性检测测试...")
        consistency_result = measure_consistency_check_performance(test_text, runs_per_test, model_name, base_url)
        results["consistency_check"].append(consistency_result)
        
        print(f"    语法纠错 - 平均耗时: {grammar_result.avg_time:.2f}s, "
              f"一致性检测 - 平均耗时: {consistency_result.avg_time:.2f}s")
    
    return results

def print_detailed_results(results: Dict[str, List[PerformanceResult]]):
    """
    打印详细的性能测试结果
    
    Args:
        results: 性能测试结果
    """
    print("\n" + "="*80)
    print("详细性能测试结果")
    print("="*80)
    
    for task_name, task_results in results.items():
        print(f"\n{task_name.upper().replace('_', ' ')}:")
        print("-" * 60)
        print(f"{'词数':<8} {'字符数':<10} {'总耗时(s)':<12} {'平均耗时(s)':<12} {'标准差':<10} {'最短(s)':<10} {'最长(s)':<10} {'响应时间(秒/千字)':<15}")
        print("-" * 60)
        
        for result in task_results:
            chars_per_k_word = (result.text_length / result.num_words) * 1000 if result.num_words > 0 else 0
            response_time_per_k_char = (result.avg_time / result.text_length) * 1000 if result.text_length > 0 else 0
            
            print(f"{result.num_words:<8} {result.text_length:<10} {result.total_time:<12.2f} "
                  f"{result.avg_time:<12.2f} {result.std_dev:<10.2f} {result.min_time:<10.2f} "
                  f"{result.max_time:<10.2f} {response_time_per_k_char*1000:<15.2f}")

def print_summary(results: Dict[str, List[PerformanceResult]]):
    """
    打印性能测试摘要
    
    Args:
        results: 性能测试结果
    """
    print("\n" + "="*80)
    print("性能测试摘要")
    print("="*80)
    
    for task_name, task_results in results.items():
        print(f"\n{task_name.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        avg_response_times = []
        for result in task_results:
            response_time_per_k_char = (result.avg_time / result.text_length) * 1000 if result.text_length > 0 else 0
            avg_response_times.append(response_time_per_k_char * 1000)  # 转换为秒/千字
        
        print(f"平均响应时间范围: {min(avg_response_times):.2f} - {max(avg_response_times):.2f} 秒/千字")
        print(f"总体平均响应时间: {statistics.mean(avg_response_times):.2f} 秒/千字")
        print(f"标准差: {statistics.stdev(avg_response_times) if len(avg_response_times) > 1 else 0:.2f} 秒/千字")

def save_results_to_file(results: Dict[str, List[PerformanceResult]], output_file: str):
    """
    将结果保存到JSON文件
    
    Args:
        results: 性能测试结果
        output_file: 输出文件路径
    """
    serializable_results = {}
    
    for task_name, task_results in results.items():
        serializable_results[task_name] = []
        for result in task_results:
            result_dict = {
                "text_length": result.text_length,
                "num_words": result.num_words,
                "total_time": result.total_time,
                "avg_time": result.avg_time,
                "std_dev": result.std_dev,
                "min_time": result.min_time,
                "max_time": result.max_time,
                "runs": result.runs,
                "response_time_per_k_char": (result.avg_time / result.text_length) * 1000 if result.text_length > 0 else 0,
                "response_time_per_k_word": (result.avg_time / result.num_words) * 1000 if result.num_words > 0 else 0
            }
            serializable_results[task_name].append(result_dict)
    output_path = os.path.join(output_file, "performance_benchmark_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到 {output_file}")

def plot_performance_results(results: Dict[str, List[PerformanceResult]], save_path: str = "logs/performance_results/"):
    """
    绘制性能测试曲线图：耗时随文本长度变化
    Args:
        results: 性能测试结果
        save_path: 图片保存路径
    """
    
    # 提取数据
    lengths = [res.text_length for res in results["grammar_check"]]
    grammar_times = [res.avg_time for res in results["grammar_check"]]
    consistency_times = [res.avg_time for res in results["consistency_check"]]
    
    # 创建画布
    plt.figure(figsize=(10, 6))
    
    # 绘制两条曲线
    plt.plot(lengths, grammar_times, marker='o', linewidth=2, label="Grammar Check", color="#1f77b4")
    plt.plot(lengths, consistency_times, marker='s', linewidth=2, label="Consistency Check", color="#ff7f0e")
    
    # 图表样式
    plt.title("Comparison of Grammar Check and Consistency Check Performance with Words Length Increasing",  fontsize=14, pad=20)
    plt.xlabel("Words Length", fontsize=12)
    plt.ylabel("Average Response Time (s)", fontsize=12)
    plt.legend(fontsize=12)  # 图例
    plt.grid(True, alpha=0.3)  # 网格
    plt.tight_layout()  # 自适应布局
    
    # 保存 + 显示
    save_path = os.path.join(save_path, "performance_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n性能曲线图已保存至: {save_path}")
    plt.show()

def plot_performance_from_logs(json_file_path: str, save_path: str = "logs/performance_results/"):
    """
    从JSON文件读取性能数据并绘制图表
    
    Args:
        json_file_path: JSON文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"错误: 文件 {json_file_path} 不存在")
        return
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取数据
    grammar_results = data.get("grammar_check", [])
    consistency_results = data.get("consistency_check", [])
    
    if not grammar_results or not consistency_results:
        print("警告: JSON文件中没有找到有效的性能数据")
        return
    
    text_lengths = [result["text_length"] for result in grammar_results]
    grammar_avg_times = [result["avg_time"] for result in grammar_results]
    consistency_avg_times = [result["avg_time"] for result in consistency_results]
    
    # 创建图表
    plt.figure(figsize=(12, 8))
    
    # 绘制语法纠错和一致性检测的平均耗时
    plt.plot(text_lengths, grammar_avg_times, marker='o', label='Grammar Check', linewidth=2, markersize=8)
    plt.plot(text_lengths, consistency_avg_times, marker='s', label='Consistency Check', linewidth=2, markersize=8)
    
    plt.xlabel('Words Length', fontsize=12)
    plt.ylabel('Average Response Time (seconds)', fontsize=12)
    plt.title('Comparison of Grammar Check and Consistency Check Performance with Words Length Increasing', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 在每个点上显示具体的数值
    for i, (x, y) in enumerate(zip(text_lengths, grammar_avg_times)):
        plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    for i, (x, y) in enumerate(zip(text_lengths, consistency_avg_times)):
        plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(save_path, "performance_curve.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n性能对比图已从 {json_file_path} 生成并保存到 {save_path}")
    
    # 显示图表
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="性能基准测试脚本")
    parser.add_argument("--lengths", nargs='+', type=int, default=[200, 500, 1000, 2000, 5000],
                       help="要测试的文本长度列表（词数），默认为 [200, 500, 1000, 2000, 5000]")
    parser.add_argument("--runs", type=int, default=3,
                       help="每个测试运行的次数，默认为3")
    parser.add_argument("--output", type=str, default="logs/performance_results/",
                       help="输出结果文件名")
    parser.add_argument("--model-name", type=str, default="gpt-5-mini",
                       help="使用的模型名称，默认为 qwen-plus")
    parser.add_argument("--base-url", type=str, default="https://api.chatanywhere.tech/v1",
                       help="模型服务的基础URL，默认为阿里云接口")
    
    args = parser.parse_args()
    
    print("开始性能基准测试...")
    print(f"测试文本长度: {args.lengths}")
    print(f"每个长度运行次数: {args.runs}")
    print(f"模型: {args.model_name}")
    print(f"基础URL: {args.base_url}")
    

    # 运行性能测试
    results = run_performance_tests(args.lengths, args.runs, args.model_name, args.base_url)
    
    # 打印详细结果
    print_detailed_results(results)
    
    # 打印摘要
    print_summary(results)
    
    # 保存结果到文件
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    save_results_to_file(results, args.output)

    plot_performance_results(results, args.output)

if __name__ == "__main__":
    #plot_performance_from_logs("logs/performance_results/performance_benchmark_results.json", "logs/performance_results/")
    main()