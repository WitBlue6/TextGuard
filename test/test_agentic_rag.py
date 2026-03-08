import os
import sys
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic.router import get_agentic_router
from agentic.indexer import AdvancedIndexer, get_query_transformer
from agentic.retriever import IterativeRetriever
from agentic.evaluator import SelfEvaluator
from filereader.reader import extract_text_from_docx, chunking

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/test_agentic_rag.log')
    ]
)
logger = logging.getLogger(__name__)

class TestAgenticRAG:
    """测试Agentic RAG功能"""
    
    def __init__(self, model_name="qwen-max", base_url="https://apis.iflow.cn/v1", embedding_name="text-embedding-v4"):
        """
        初始化测试类
        :param model_name: 模型名称
        :param base_url: API基础URL
        :param embedding_name: 嵌入模型名称
        """
        self.model_name = model_name
        self.base_url = base_url
        self.embedding_name = embedding_name
        self.test_results = {}
    
    def test_router(self):
        """测试智能路由决策器"""
        logger.info("=== 测试智能路由决策器 ===")
        
        router = get_agentic_router(self.model_name, self.base_url)
        
        test_queries = [
            "什么是人工智能？",  # 应该选择 direct
            "TextGuard系统的主要功能是什么？",  # 应该选择 retrieve
            "如何使用TextGuard进行文本一致性检查并修复错误？",  # 应该选择 decompose
        ]
        
        results = []
        for query in test_queries:
            try:
                decision = router.invoke({"query": query}).content.strip()
                results.append({"query": query, "decision": decision})
                logger.info(f"查询: {query}\n决策: {decision}")
            except Exception as e:
                logger.error(f"路由决策失败: {e}")
        
        self.test_results["router"] = results
        return results
    
    def test_query_transformer(self):
        """测试查询转换器"""
        logger.info("=== 测试查询转换器 ===")
        
        transformer = get_query_transformer(self.model_name, self.base_url)
        
        test_query = "TextGuard系统的实体一致性检查功能如何工作？"
        
        results = {}
        
        # 测试多查询生成
        try:
            multi_queries = transformer["multi_query"].invoke({"query": test_query}).content
            results["multi_query"] = multi_queries.split('\n')
            logger.info("多查询生成结果:")
            for i, q in enumerate(results["multi_query"]):
                if q.strip():
                    logger.info(f"  {i+1}. {q.strip()}")
        except Exception as e:
            logger.error(f"多查询生成失败: {e}")
        
        # 测试HyDE
        try:
            hyde = transformer["hyde"].invoke({"query": test_query}).content
            results["hyde"] = hyde
            logger.info("HyDE生成结果:")
            logger.info(f"  {hyde[:100]}...")
        except Exception as e:
            logger.error(f"HyDE生成失败: {e}")
        
        # 测试查询分解
        try:
            decomposed = transformer["decompose"].invoke({"query": test_query}).content
            results["decompose"] = decomposed.split('\n')
            logger.info("查询分解结果:")
            for i, q in enumerate(results["decompose"]):
                if q.strip():
                    logger.info(f"  {i+1}. {q.strip()}")
        except Exception as e:
            logger.error(f"查询分解失败: {e}")
        
        self.test_results["query_transformer"] = results
        return results
    
    def test_indexer(self):
        """测试高级索引器"""
        logger.info("=== 测试高级索引器 ===")
        
        indexer = AdvancedIndexer(self.embedding_name, self.base_url)
        
        # 测试多表示创建
        test_text = "TextGuard是一个中文文本错误检测系统，主要功能包括语法检查、实体提取、实体一致性检查和文本修正。系统使用LLM进行智能分析，能够识别和修复文本中的各种错误。"
        
        try:
            multi_reps = indexer.create_multi_representation(test_text)
            logger.info(f"多表示创建成功，生成了 {len(multi_reps)} 个表示")
            for i, rep in enumerate(multi_reps):
                logger.info(f"  表示 {i+1}: {rep[:100]}...")
        except Exception as e:
            logger.error(f"多表示创建失败: {e}")
        
        # 测试Raptor索引创建
        test_texts = [
            "TextGuard的语法检查功能可以识别中文文本中的语法错误。",
            "实体提取功能可以从文本中提取各类实体。",
            "实体一致性检查可以发现文本中实体的一致性问题。",
            "文本修正功能可以根据检查结果修正文本错误。",
            "系统使用LLM进行智能分析，提高检测准确性。"
        ]
        
        try:
            persist_dir = f"./test/test_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            raptor_index = indexer.create_raptor_index(test_texts, persist_directory=persist_dir)
            logger.info(f"Raptor索引创建成功，保存路径: {persist_dir}")
        except Exception as e:
            logger.error(f"Raptor索引创建失败: {e}")
        
        self.test_results["indexer"] = "测试完成"
        return "测试完成"
    
    def test_retriever(self):
        """测试迭代检索器"""
        logger.info("=== 测试迭代检索器 ===")
        
        # 创建测试索引
        indexer = AdvancedIndexer(self.embedding_name, self.base_url)
        test_texts = [
            "TextGuard的语法检查功能可以识别中文文本中的语法错误，包括主谓不一致、成分残缺等问题。",
            "实体提取功能可以从文本中提取各类实体，如人物、组织、地点等。",
            "实体一致性检查可以发现文本中实体的一致性问题，如同一实体的属性冲突。",
            "文本修正功能可以根据检查结果修正文本错误，保持原意的同时提高文本质量。",
            "系统使用LLM进行智能分析，提高检测准确性和修正质量。"
        ]
        
        try:
            persist_dir = f"./test/test_retriever_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            vectorstore = indexer.create_raptor_index(test_texts, persist_directory=persist_dir)
        except Exception as e:
            logger.error(f"Raptor索引创建失败: {e}，已使用简单索引")

        try:
            # 创建迭代检索器
            retriever = IterativeRetriever(vectorstore, self.model_name, self.base_url, self.embedding_name)
            
            # 测试迭代检索
            test_query = "TextGuard如何检测和修正文本中的语法错误？"
            results = retriever.retrieve(test_query, max_iterations=3)
            
            logger.info(f"迭代检索完成，共进行了 {len(results)} 次迭代")
            for i, docs in enumerate(results):
                logger.info(f"  第 {i+1} 次检索结果:")
                for j, doc in enumerate(docs):
                    logger.info(f"    文档 {j+1}: {doc[:1000]}...")
            
            self.test_results["retriever"] = "测试完成"
            return "测试完成"
            
        except Exception as e:
            logger.error(f"迭代检索测试失败: {e}")
            return f"测试失败: {e}"
    
    def test_evaluator(self):
        """测试自我评估器"""
        logger.info("=== 测试自我评估器 ===")
        
        evaluator = SelfEvaluator(self.model_name, self.base_url)
        
        # 测试检索结果评估
        test_query = "TextGuard的语法检查功能"
        test_docs = ['TextGuard的语法检查功能可以识别中文文本中的语法错误，包括主谓不一致、成分残缺等问题。', '系统使用LLM进行智能分析，提高检测准确性和修正质量。']
        
        try:
            retrieval_eval = evaluator.evaluate_retrieval(test_query, test_docs)
            logger.info("检索结果评估:")
            logger.info(f"  {retrieval_eval}")
        except Exception as e:
            logger.error(f"检索结果评估失败: {e}")
        
        # 测试回答质量评估
        test_answer = "TextGuard的语法检查功能可以识别中文文本中的语法错误，包括主谓不一致、成分残缺等问题，使用LLM进行智能分析提高检测准确性。"
        
        try:
            answer_eval = evaluator.evaluate_answer(test_query, test_answer, test_docs)
            logger.info("回答质量评估:")
            logger.info(f"  {answer_eval}")
        except Exception as e:
            logger.error(f"回答质量评估失败: {e}")
        
        self.test_results["evaluator"] = "测试完成"
        return "测试完成"
    
    def test_integration(self):
        """测试Agentic RAG集成功能"""
        logger.info("=== 测试Agentic RAG集成功能 ===")
        
        try:
            # 1. 加载测试文件
            test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "test.docx")
            if not os.path.exists(test_file):
                test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "test.pdf")
            
            if os.path.exists(test_file):
                text = extract_text_from_docx(test_file)
                chunks = chunking(text, chunk_size=1000)
                logger.info(f"加载测试文件成功，分块数量: {len(chunks)}")
            else:
                logger.warning("未找到测试文件，使用示例文本")
                text = "TextGuard是一个中文文本错误检测系统，主要功能包括语法检查、实体提取、实体一致性检查和文本修正。系统使用LLM进行智能分析，能够识别和修复文本中的各种错误。"
                chunks = [text]
            
            # 2. 创建索引
            indexer = AdvancedIndexer(self.embedding_name, self.base_url)
            all_texts = []
            for chunk in chunks:
                multi_reps = indexer.create_multi_representation(chunk)
                all_texts.extend(multi_reps)
            
            persist_dir = f"./test_integration_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            vectorstore = indexer.create_raptor_index(all_texts, persist_directory=persist_dir)
            
            # 3. 创建路由和检索器
            router = get_agentic_router(self.model_name, self.base_url)
            retriever = IterativeRetriever(vectorstore, self.model_name, self.base_url, self.embedding_name)
            evaluator = SelfEvaluator(self.model_name, self.base_url)
            
            # 4. 测试完整流程
            test_queries = [
                "TextGuard系统的主要功能是什么？",
                "如何使用TextGuard进行文本一致性检查？",
                "TextGuard的语法检查功能如何工作？"
            ]
            
            for query in test_queries:
                logger.info(f"\n测试查询: {query}")
                
                # 路由决策
                decision = router.invoke({"query": query}).content.strip()
                logger.info(f"路由决策: {decision}")
                
                if "retrieve" in decision.lower():
                    # 迭代检索
                    results = retriever.retrieve(query)
                    logger.info(f"检索完成，共 {len(results)} 次迭代")
                    
                    # 评估检索结果
                    eval_result = evaluator.evaluate_retrieval(query, results[-1])
                    logger.info(f"检索结果评估: {eval_result}")
                
            logger.info("集成测试完成")
            self.test_results["integration"] = "测试完成"
            return "测试完成"
            
        except Exception as e:
            logger.error(f"集成测试失败: {e}", exc_info=True)
            return f"测试失败: {e}"
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("开始运行Agentic RAG测试")
        
        tests = [
            ("router", self.test_router),
            ("query_transformer", self.test_query_transformer),
            ("indexer", self.test_indexer),
            ("retriever", self.test_retriever),
            ("evaluator", self.test_evaluator),
            ("integration", self.test_integration)
        ]
        
        for test_name, test_func in tests:
            logger.info(f"\n========== 运行测试: {test_name} ==========")
            try:
                test_func()
                logger.info(f"测试 {test_name} 完成")
            except Exception as e:
                logger.error(f"测试 {test_name} 失败: {e}")
        
        logger.info("\n所有测试完成")
        return self.test_results

if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    import os
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="测试Agentic RAG功能")
    parser.add_argument("--model_name", type=str, default="qwen3-max", help="模型名称")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API基础URL")
    parser.add_argument("--embedding_name", type=str, default="text-embedding-v4", help="Embedding model name")

    args = parser.parse_args()
    
    # 确保 base_url 以 /v1 结尾
    if not args.base_url.endswith("/v1"):
        args.base_url = args.base_url.rstrip("/") + "/v1"
    
    # 运行测试
    tester = TestAgenticRAG(model_name=args.model_name, base_url=args.base_url, embedding_name=args.embedding_name)
    results = tester.run_all_tests()
    
    # 打印测试结果
    logger.info("\n测试结果汇总:")
    for test_name, result in results.items():
        logger.info(f"{test_name}: {result}")