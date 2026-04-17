"""
本地向量模型索引器
使用本地模型进行文本嵌入
"""
import chromadb
import numpy as np
import logging
from typing import List, Optional
from sklearn.cluster import KMeans
import os

logger = logging.getLogger(__name__)


class LocalAdvancedIndexer:
    """
    本地向量索引器
    支持本地模型进行文本嵌入和Raptor索引创建
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_model_path: Optional[str] = None,
                 device: Optional[str] = None):
        """
        初始化本地向量索引器

        :param embedding_model: 本地嵌入模型名称或路径
        :param model_path: 本地模型路径（用于sentence-transformers）
        :param device: 设备 ('cuda', 'cpu', 'mps')
        """
        logger.info(f"初始化本地向量索引器，模型: {embedding_model}")

        # 设置设备
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            device = "cpu" if device == "cuda" else device
        self.device = device

        logger.info(f"使用设备: {device}")

        # 加载嵌入模型
        try:
            from sentence_transformers import SentenceTransformer
            if embedding_model_path:
                self.encoder = SentenceTransformer(embedding_model_path, device=device)
                logger.info(f"从本地路径加载嵌入模型: {embedding_model_path}")
            else:
                self.encoder = SentenceTransformer(embedding_model, device=device)
                logger.info(f"从HuggingFace加载嵌入模型: {embedding_model}")

        except ImportError:
            logger.error("sentence-transformers 未安装，请运行: pip install sentence-transformers")
            raise

        self.text_splitter = None  # 将在create_raptor_index中定义

    def create_raptor_index_local(self, texts: List[str],
                                   persist_directory: str = "./chroma_db_local",
                                   chunk_size: int = 1000,
                                   chunk_overlap: int = 200,
                                   num_clusters: int = None) -> chromadb.Collection:
        """
        创建本地Raptor索引（分层摘要+聚类）

        :param texts: 要索引的文本列表
        :param persist_directory: 向量存储的持久化目录
        :param chunk_size: 文本块大小
        :param chunk_overlap: 文本块重叠大小
        :param num_clusters: 聚类数量，None则自动计算
        :return: ChromaDB集合
        """
        try:
            logger.info(f"开始创建本地Raptor索引，文本数量: {len(texts)}")

            # 设置文本分割器
            if self.text_splitter is None:
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

            # 分割文本
            all_chunks = []
            for text in texts:
                chunks = self.text_splitter.split_text(text)
                all_chunks.extend(chunks)

            logger.info(f"文本分割完成，总chunk数: {len(all_chunks)}")

            # 批量生成嵌入
            batch_size = 10
            embeddings = []
            total_chunks = len(all_chunks)

            logger.info(f"开始批量生成嵌入，批次大小: {batch_size}")

            for i in range(0, total_chunks, batch_size):
                batch = all_chunks[i:i+batch_size]
                logger.debug(f"处理批次 {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")

                try:
                    batch_embeddings = self.encoder.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False
                    )
                    embeddings.extend(batch_embeddings.tolist())

                except Exception as e:
                    logger.error(f"批次 {i//batch_size + 1} 嵌入失败: {e}")
                    # 失败时使用空向量
                    batch_embeddings = np.zeros((len(batch), self.encoder.get_sentence_embedding_dimension()))
                    embeddings.extend(batch_embeddings.tolist())

            logger.info(f"嵌入生成完成，向量维度: {len(embeddings[0]) if embeddings else 0}")

            # 自动计算聚类数量
            if num_clusters is None:
                num_clusters = max(1, min(5, len(all_chunks) // 2))
            logger.info(f"聚类数量: {num_clusters}")

            # 聚类
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            logger.info(f"聚类完成")

            # 为每个聚类生成总结
            cluster_summaries = []
            cluster_texts = []

            for i in range(num_clusters):
                cluster_texts.append([all_chunks[j] for j, c in enumerate(clusters) if c == i])

            for i, texts in enumerate(cluster_texts):
                if len(texts) == 1:
                    summary = texts[0]
                else:
                    combined_text = "\n".join(texts)
                    summary = self._generate_summary_local(combined_text)

                    if not summary:
                        summary = texts[0]

                cluster_summaries.append(summary)

            logger.info(f"聚类总结生成完成")

            # 为聚类总结生成嵌入
            cluster_embeddings = self.encoder.encode(
                cluster_summaries,
                convert_to_numpy=True,
                show_progress_bar=False
            ).tolist()

            # 创建ChromaDB集合
            chromadb_client = chromadb.PersistentClient(path=persist_directory)
            collection = chromadb_client.get_or_create_collection(name="raptor_local")

            # 添加数据
            collection.add(
                documents=cluster_summaries,
                embeddings=cluster_embeddings,
                ids=[f"cluster_{i}" for i in range(len(cluster_summaries))]
            )

            logger.info(f"本地Raptor索引创建完成，包含 {len(cluster_summaries)} 个聚类")

            return collection

        except Exception as e:
            logger.error(f"创建本地Raptor索引失败: {e}")
            raise

    def read_index_local(self, persist_directory: str = "./chroma_db_local") -> chromadb.Collection:
        """
        读取已存在的本地Raptor索引

        :param persist_directory: 向量存储的持久化目录
        :return: ChromaDB集合
        """
        try:
            chromadb_client = chromadb.PersistentClient(path=persist_directory)
            collection = chromadb_client.get_or_create_collection(name="raptor_local")
            logger.info(f"本地Raptor索引读取完成")
            return collection
        except Exception as e:
            logger.error(f"读取本地Raptor索引失败: {e}")
            return None

    def _generate_summary_local(self, text: str, max_length: int = 200) -> str:
        """
        生成文本摘要（本地版本）

        :param text: 输入文本
        :param max_length: 最大摘要长度
        :return: 文本摘要
        """
        try:
            if len(text) <= max_length:
                return text
            else:
                return text[:max_length] + "..."
        except Exception as e:
            logger.error(f"生成摘要失败: {e}")
            return ""

    def add_documents_local(self, texts: List[str],
                            ids: Optional[List[str]] = None,
                            persist_directory: str = "./chroma_db_local"):
        """
        添加文档到索引（本地版本）

        :param texts: 文档列表
        :param ids: 文档ID列表
        :param persist_directory: 持久化目录
        :return: 添加结果
        """
        try:
            chromadb_client = chromadb.PersistentClient(path=persist_directory)
            collection = chromadb_client.get_or_create_collection(name="raptor_local")

            if ids is None:
                ids = [f"doc_{i}" for i in range(len(texts))]

            # 批量生成嵌入
            batch_size = 10
            embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                batch_embeddings = self.encoder.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False
                ).tolist()
                embeddings.extend(batch_embeddings)

            # 添加文档
            collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=ids
            )

            logger.info(f"成功添加 {len(texts)} 个文档到本地索引")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    def query_local(self, query: str, n_results: int = 5,
                    persist_directory: str = "./chroma_db_local") -> dict:
        """
        查询本地索引

        :param query: 查询文本
        :param n_results: 返回结果数量
        :param persist_directory: 持久化目录
        :return: 查询结果
        """
        try:
            chromadb_client = chromadb.PersistentClient(path=persist_directory)
            collection = chromadb_client.get_or_create_collection(name="raptor_local")

            # 生成查询嵌入
            query_embedding = self.encoder.encode(
                query,
                convert_to_numpy=True
            ).tolist()

            # 执行查询
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            return {
                "documents": results['documents'][0] if results['documents'] else [],
                "distances": results['distances'][0] if results['distances'] else []
            }

        except Exception as e:
            logger.error(f"查询本地索引失败: {e}")
            return {"documents": [], "distances": []}
