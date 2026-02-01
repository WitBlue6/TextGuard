from llm.model import get_entity_extract_chain, get_entity_consistency_check_chain, get_memory_summary_chain, get_consistency_correct_chain
from llm.entity import extract_entities, check_entity_consistency, summarize_entity_memory
from llm.entity import EntityStore
from filereader.reader import extract_text_from_pdf, extract_text_from_docx, chunking
# Agentic RAG imports
from agentic.router import get_agentic_router
from agentic.indexer import AdvancedIndexer, get_query_transformer
from agentic.retriever import IterativeRetriever
from agentic.evaluator import SelfEvaluator


import argparse
import logging
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Consistency Check Model")
    parser.add_argument("--model_name", type=str, default="qwen-plus", help="Model name")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="Base URL")
    parser.add_argument("--docx_data", type=str, default="./dataset/test_long.docx", help="Docs Dataset path")
    #parser.add_argument("--pdf_data", type=str, default="./dataset/test.pdf", help="PDF Dataset path")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Output path")
    parser.add_argument("--chroma_db_dir", type=str, default="./chroma_db", help="ChromaDB Save Path")
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

def check_consistency(args, **kwargs):
    '''
    检查文档中的实体一致性
    返回冲突检测结果列表
    '''
    logger = kwargs.get("logger")

    # chain获取
    entity_extract_chain = get_entity_extract_chain(args.model_name, args.base_url)
    entity_consistency_check_chain = get_entity_consistency_check_chain(args.model_name, args.base_url)
    memory_summary_chain = get_memory_summary_chain(args.model_name, args.base_url)

    # 初始化Agentic RAG组件
    logger.info("初始化Agentic RAG组件...")
    agentic_router = get_agentic_router(args.model_name, args.base_url)
    query_transformer = get_query_transformer(args.model_name, args.base_url)
    advanced_indexer = AdvancedIndexer(args.model_name, args.base_url)
    self_evaluator = SelfEvaluator(args.model_name, args.base_url)
    
    raptor_index = advanced_indexer.read_index(persist_directory=args.chroma_db_dir)
    
    iterative_retriever = IterativeRetriever(raptor_index, args.model_name, args.base_url)
 
    # 文档读取
    logger.info(f"读取文档: {args.docx_data}")
    text = extract_text_from_docx(args.docx_data)
    chunks = chunking(text)

    # chunking后，保留上下文提取实体
    ent_store = EntityStore()
    previous_memory = ""

    for i, chunk in enumerate(chunks):
        chunk_input = f"前文要点总结:{previous_memory}\n当前输入文本:{chunk}" if previous_memory else chunk
        # 提出本chunk的实体
        ents = extract_entities(entity_extract_chain, chunk_input)
        for ent in ents:
            ent_store.add_entity(ent)
        logger.info(f"当前chunk实体: {ents}")

        if i < len(chunks) - 1:
            # 更新memory，用于下一chunk
            previous_memory = summarize_entity_memory(memory_summary_chain, chunk_input)
            logger.info(f"对第{i+1}个chunk总结: {previous_memory}")

    # # 不chunking，直接提取所有实体
    # logger.info("开始提取所有实体")
    # ents = extract_entities(entity_extract_chain, text)
    # for ent in ents:
    #     ent_store.add_entity(ent)
    # logger.info(f"所有实体: {ents}")

    # 冲突检测 - 融合RAG
    consistency_results = []
    logger.info(f"提取出的所有实体: {ent_store.all_entities()}")
    logger.info("开始检测实体冲突")
    for ent in ent_store.all_entities():
        entity_description = f"实体名称: {ent.name}\n实体类型: {ent.type}\n实体属性: {ent.attributes}\n实体事件: {ent.events}\n实体关系: {ent.relations}"
        
        # 智能路由决策
        route_query = f"是否需要检索更多信息来检查以下实体的一致性？\n{entity_description}"
        route_decision = agentic_router.invoke({"query": route_query}).content
        logger.info(f"路由决策: {route_decision}")
        if "retrieve" in route_decision.lower() and iterative_retriever:
            # 使用RAG进行检索增强一致性检查
            logger.info(f"使用RAG检索增强实体 {ent.entity_id} 的一致性检查")   
            # 查询转换
            transformed_queries = query_transformer["multi_query"].invoke({"query": route_query}).content  
            logger.info(f"转换后的查询: {transformed_queries}")
            # 迭代检索
            retrieval_results = iterative_retriever.retrieve(transformed_queries) 
            # 使用检索结果增强一致性检查
            enhanced_input = f"实体信息: {entity_description}\n\n检索到的相关信息: {retrieval_results[-1]}\n\n请检查该实体的一致性"
            
            res = check_entity_consistency(entity_consistency_check_chain, ent, enhanced_input)
        else:
            # 直接检查一致性
            res = check_entity_consistency(entity_consistency_check_chain, ent)
        
        logger.info(f"对于实体 {ent.entity_id} 的冲突分析: {res}")
        consistency_results.append(res)

    # 保存一致性检查结果
    consistency_save_name = kwargs.get("save_name", "consistency_result.json")
    save_dir = os.path.join(args.log_dir, os.path.basename(args.docx_data).split(".")[0])
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, consistency_save_name), "w", encoding="utf-8") as f:
        json.dump(consistency_results, f, ensure_ascii=False, indent=4)
        logger.info(f"一致性检查结果已保存到: {os.path.join(save_dir, consistency_save_name)}")
    # 保留全部实体列表
    all_entities_save_name = "all_entities.json"
    with open(os.path.join(save_dir, all_entities_save_name), "w", encoding="utf-8") as f:
        json.dump([ent.model_dump() for ent in ent_store.all_entities()], f, ensure_ascii=False, indent=4)
        logger.info(f"所有实体已保存到: {os.path.join(save_dir, all_entities_save_name)}")

    return consistency_results

def correct_based_on_consistency(args, **kwargs):
    """
    根据一致性检查结果，标记长文本中的实体冲突
    返回标记后的长文本
    """
    consistency_results = kwargs.get("consistency_results")
    logger = kwargs.get("logger")

    # 对输入的实体进行剔除，只保留冲突实体
    conflict_ents = [ent for ent in consistency_results if ent["has_conflict"] is True]
    logger.info(f"冲突实体: {conflict_ents}")
    consistency_correct_chain = get_consistency_correct_chain(args.model_name, args.base_url)

    # 文档读取
    logger.info(f"读取文档: {args.docx_data}")
    text = extract_text_from_docx(args.docx_data)
    chunks = chunking(text)
    res_list = []

    # 对每个chunk进行修正
    for chunk in chunks:
        chunk_input = f"原始文本:{chunk}\n实体冲突分析结果:{consistency_results}"
        res = consistency_correct_chain.invoke(chunk_input).content
        logger.info(f"段落修正结果: \n{res}")
        res_dict = {
            "original_text": chunk,
            "corrected_text": res
        }
        res_list.append(res_dict)

    # 保存修正后的结果为txt文件
    save_name = kwargs.get("save_name", "corrected_result.txt")
    save_dir = os.path.join(args.log_dir, os.path.basename(args.docx_data).split(".")[0])
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, save_name), "w", encoding="utf-8") as f:
        json.dump(res_list, f, ensure_ascii=False, indent=4)
        logger.info(f"修正后的结果已保存到: {os.path.join(save_dir, save_name)}")
    
    return res_list

def get_consistency_from_file(args, **kwargs):
    """
    从文件中读取一致性检查结果
    返回一致性检查结果列表
    """
    consistency_save_name = kwargs.get("save_name", "consistency_result.json")
    save_dir = os.path.join(args.log_dir, os.path.basename(args.docx_data).split(".")[0])
    logger = kwargs.get("logger")
    logger.info(f"从文件 {os.path.join(save_dir, consistency_save_name)} 读取一致性检查结果")
    with open(os.path.join(save_dir, consistency_save_name), "r", encoding="utf-8") as f:
        consistency_results = json.load(f)
    return consistency_results

if __name__ == "__main__":
    args = parse_args()
    logger = logging_config(args)

    logger.info(f"开始运行一致性检查，模型: {args.model_name}, 数据集: {args.docx_data}")

    consistency = check_consistency(args, logger=logger)
    logger.info(f"一致性检查结果: {consistency}")

    # # 从文件中读取一致性检查结果
    # consistency = get_consistency_from_file(args, logger=logger)

    corrected_chunks = correct_based_on_consistency(args, consistency_results=consistency, logger=logger)
    logger.info(f"修正后的chunk结果: {corrected_chunks}")
