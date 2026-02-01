from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import logging

logger = logging.getLogger(__name__)

class IterativeRetriever:
    def __init__(self, vectorstore, model_name: str = "gpt-4o-mini-2024-07-18", base_url: str = "https://free.v36.cm/v1"):
        """
        初始化迭代检索器
        
        :param vectorstore: 向量存储实例
        :param model_name: 使用的LLM模型名称
        :param base_url: LLM API的基础URL
        """
        self.vectorstore = vectorstore
        self.model = ChatOpenAI(
            model_name=model_name,
            temperature=0.7,
            max_tokens=1000,
            base_url=base_url,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    
    def retrieve(self, query: str, max_iterations: int = 3, k: int = 3):
        """
        执行迭代检索，根据需要多次优化查询
        
        :param query: 用户查询
        :param max_iterations: 最大迭代次数
        :param k: 每次检索返回的文档数量
        :return: 检索结果列表
        """
        try:
            current_query = query
            all_results = []
            
            for i in range(max_iterations):
                logger.info(f"迭代检索第 {i+1} 次，查询: {current_query}")
                # 检查是否为空
                if not self.vectorstore:
                    logger.warning("向量存储未初始化，无法执行检索")
                    break
                # 执行检索
                docs = self.vectorstore.similarity_search(current_query, k=k)
                all_results.append(docs)
                
                # 评估检索结果是否足够
                if self._is_retrieval_sufficient(docs, current_query):
                    logger.info(f"检索结果足够，停止迭代")
                    break
                
                # 重写查询
                new_query = self._rewrite_query(current_query, docs)
                if new_query == current_query:
                    logger.info(f"查询无法进一步优化，停止迭代")
                    break
                
                current_query = new_query
            
            return all_results
            
        except Exception as e:
            logger.error(f"迭代检索失败: {e}")
            # 失败时执行简单检索
            return [self.vectorstore.similarity_search(query, k=k)]
    
    def _is_retrieval_sufficient(self, docs, query):
        """
        评估检索结果是否足够回答查询
        
        :param docs: 检索到的文档
        :param query: 用户查询
        :return: 布尔值，表示结果是否足够
        """
        try:
            eval_prompt = ChatPromptTemplate.from_messages([
                ("system", "评估检索结果是否足够回答查询。回答'是'或'否'，不要添加任何其他内容。"),
                ("human", "查询：{query}\n\n检索结果：{docs}\n\n是否足够？"),
            ])
            
            # 构建文档字符串
            docs_str = "\n".join([f"文档 {i+1}: {doc.page_content[:200]}..." for i, doc in enumerate(docs)])
            
            result = self.model.invoke(eval_prompt.format(query=query, docs=docs_str)).content.strip()
            logger.debug(f"检索结果评估: {result}")
            
            return "是" in result
            
        except Exception as e:
            logger.error(f"评估检索结果失败: {e}")
            return True  # 失败时默认结果足够
    
    def _rewrite_query(self, query, docs):
        """
        根据检索结果重写查询，使其更精确
        
        :param query: 原始查询
        :param docs: 检索到的文档
        :return: 重写后的查询
        """
        try:
            rewrite_prompt = ChatPromptTemplate.from_messages([
                ("system", "根据检索结果重写查询，使其更精确，更能找到相关信息。只输出重写后的查询，不要添加任何解释。"),
                ("human", "原始查询：{query}\n\n检索结果：{docs}\n\n重写后的查询："),
            ])
            
            # 构建文档字符串
            docs_str = "\n".join([f"文档 {i+1}: {doc.page_content[:200]}..." for i, doc in enumerate(docs)])
            
            result = self.model.invoke(rewrite_prompt.format(query=query, docs=docs_str)).content.strip()
            logger.debug(f"查询重写: {result}")
            
            return result if result else query
            
        except Exception as e:
            logger.error(f"重写查询失败: {e}")
            return query  # 失败时返回原始查询