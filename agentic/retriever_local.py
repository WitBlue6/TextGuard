"""
本地迭代检索器
支持使用本地LLM进行查询重写和评估
"""
import chromadb
import numpy as np
import logging
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
import os

logger = logging.getLogger(__name__)


class LocalIterativeRetriever:
    """
    本地迭代检索器
    支持使用本地模型进行查询重写和结果评估
    """

    def __init__(self, vectorstore: chromadb.Collection,
                 model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 model_path: str = None,
                 device: str = None,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_model_path: str = None,
                 max_iterations: int = 3,
                 k: int = 3):
        """
        初始化本地迭代检索器

        :param vectorstore: ChromaDB集合
        :param model_name: LLM模型名称
        :param model_path: LLM模型路径
        :param device: 设备
        :param embedding_model: 嵌入模型名称
        :param embedding_model_path: 嵌入模型路径
        :param max_iterations: 最大迭代次数
        :param k: 每次检索返回的文档数量
        """
        self.chromadb_collection = vectorstore
        self.max_iterations = max_iterations
        self.k = k
        self.device = device or ("cuda" if __import__('torch').cuda.is_available() else "cpu")

        # 加载LLM模型
        logger.info(f"加载本地LLM模型: {model_name}")
        try:
            from .indexer_local import LocalAdvancedIndexer
            self.llm = LocalAdvancedIndexer(embedding_model=embedding_model, model_path=embedding_model_path, device=self.device)
        except Exception as e:
            logger.error(f"加载LLM失败: {e}")
            self.llm = None

        logger.info(f"本地迭代检索器初始化完成，max_iterations: {max_iterations}, k: {k}")

    def retrieve_local(self, query: str) -> List[str]:
        """
        执行迭代检索（本地版本）

        :param query: 用户查询
        :return: 检索结果列表
        """
        try:
            logger.info(f"开始迭代检索，查询: {query}")

            if not self.chromadb_collection:
                logger.warning("向量存储未初始化，无法执行检索")
                return []

            current_query = query
            all_results = []

            for i in range(self.max_iterations):
                logger.info(f"迭代检索第 {i+1} 次，查询: {current_query}")

                # 执行嵌入和检索
                docs = self._retrieve_with_query(current_query)

                all_results.append(docs)

                # 评估检索结果是否足够
                if self._is_retrieval_sufficient(docs, current_query):
                    logger.info(f"检索结果足够，停止迭代")
                    break

                # 如果不够，进行重写查询
                new_query = self._rewrite_query(current_query, docs)
                if new_query == current_query:
                    logger.info(f"查询无法进一步优化，停止迭代")
                    break

                current_query = new_query

            logger.info(f"迭代检索完成，共 {len(all_results)} 次迭代")
            return all_results[-1] if all_results else []

        except Exception as e:
            logger.error(f"迭代检索失败: {e}")
            # 失败时执行简单检索
            return self._simple_retrieve(query)

    def _retrieve_with_query(self, query: str) -> List[str]:
        """
        使用查询文本执行检索

        :param query: 查询文本
        :return: 检索到的文档列表
        """
        try:
            # 生成查询嵌入
            if self.llm and hasattr(self.llm, 'encoder'):
                query_embedding = self.llm.encoder.encode(
                    query,
                    convert_to_numpy=True
                ).tolist()
            else:
                # 如果没有嵌入模型，返回空
                return []

            # 执行检索
            results = self.chromadb_collection.query(
                query_embeddings=[query_embedding],
                n_results=self.k
            )

            docs = results['documents'][0] if results['documents'] else []
            return docs

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def _is_retrieval_sufficient(self, docs: List[str], query: str) -> bool:
        """
        评估检索结果是否足够回答查询（本地LLM版本）

        :param docs: 检索到的文档
        :param query: 用户查询
        :return: 布尔值，表示结果是否足够
        """
        try:
            if self.llm is None:
                # 没有LLM时，直接返回True
                return True

            # 构建评估提示
            eval_prompt = ChatPromptTemplate.from_messages([
                ("system", "评估检索结果是否足够回答查询。回答'是'或'否'，不要添加任何其他内容。"),
                ("human", "查询：{query}\n\n检索结果：{docs}\n\n是否足够？"),
            ])

            # 构建文档字符串
            docs_str = "\n".join([f"文档 {i+1}: {doc[:1000]}..." for i, doc in enumerate(docs)])

            # 调用LLM评估
            result = self.llm.invoke(
                eval_prompt.format(query=query, docs=docs_str)
            ).strip()

            logger.debug(f"检索结果评估: {result}")

            return "是" in result

        except Exception as e:
            logger.error(f"评估检索结果失败: {e}")
            return True  # 失败时默认结果足够

    def _rewrite_query(self, query: str, docs: List[str]) -> str:
        """
        根据检索结果重写查询，使其更精确（本地LLM版本）

        :param query: 原始查询
        :param docs: 检索到的文档
        :return: 重写后的查询
        """
        try:
            if self.llm is None:
                return query

            # 构建重写提示
            rewrite_prompt = ChatPromptTemplate.from_messages([
                ("system", "根据检索结果重写查询，使其更精确，更能找到相关信息。只输出重写后的查询，不要添加任何解释。"),
                ("human", "原始查询：{query}\n\n检索结果：{docs}\n\n重写后的查询："),
            ])

            # 构建文档字符串
            docs_str = "\n".join([f"文档 {i+1}: {doc[:1000]}..." for i, doc in enumerate(docs)])

            # 调用LLM重写
            result = self.llm.invoke(
                rewrite_prompt.format(query=query, docs=docs_str)
            ).strip()

            logger.debug(f"查询重写: {result}")

            return result if result else query

        except Exception as e:
            logger.error(f"重写查询失败: {e}")
            return query  # 失败时返回原始查询

    def _simple_retrieve(self, query: str) -> List[str]:
        """
        简单检索（无迭代）

        :param query: 查询文本
        :return: 检索到的文档列表
        """
        try:
            # 使用文本检索（如果ChromaDB支持）
            results = self.chromadb_collection.query(
                query_texts=[query],
                n_results=self.k
            )

            return results['documents'][0] if results['documents'] else []

        except Exception as e:
            logger.error(f"简单检索失败: {e}")
            return []

    def retrieve_with_enhancement_local(self, query: str) -> List[str]:
        """
        使用RAG增强检索结果（本地版本）

        :param query: 用户查询
        :return: 增强的检索结果列表
        """
        try:
            logger.info(f"使用RAG增强检索，查询: {query}")

            # 执行迭代检索
            retrieval_results = self.retrieve_local(query)

            if not retrieval_results:
                logger.warning("没有检索到结果")
                return []

            # 评估是否需要进一步增强
            if not self._is_retrieval_sufficient(retrieval_results, query):
                logger.info("检索结果不足，进一步检索")
                # 这里可以添加更多的增强逻辑
                # 例如：多次检索、混合检索等

            return retrieval_results

        except Exception as e:
            logger.error(f"RAG增强检索失败: {e}")
            return []
