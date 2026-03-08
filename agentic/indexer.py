from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb
import numpy as np
from sklearn.cluster import KMeans
import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class AdvancedIndexer:
    def __init__(self, embedding_name: str = "text-embedding-v4", base_url: str = None):
        """
        初始化Advanced Indexer组件
        
        :param embedding_name: 使用的Embedding模型名称
        :param base_url: Embedding API的基础URL
        """
        # 如果没有提供base_url，从环境变量获取
        if base_url is None:
            base_url = os.getenv("AI_MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        # 确保base_url以正确的路径结尾
        if not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
            
        # self.embeddings = OpenAIEmbeddings(
        #     model=embedding_name,
        #     base_url=base_url,
        #     api_key=os.getenv("OPENAI_API_KEY"),
        # )
        self.embeddings = OpenAI(
            base_url=base_url,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.embedding_model = embedding_name
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def create_multi_representation(self, text: str) -> list:
        """
        为文本创建多种表示形式，提高检索准确性
        
        :param text: 输入文本
        :return: 文本的多种表示形式列表
        """
        try:
            # 原始文本
            chunks = self.text_splitter.split_text(text)
            multi_reps = []
            
            # 为每个chunk生成多种表示
            for chunk in chunks:
                # 原始chunk
                multi_reps.append(chunk)
                
                # 实体表示 - 使用简单的关键词提取作为替代
                # 注意：这里可以替换为更复杂的实体提取逻辑
                keywords = self._extract_keywords(chunk)
                if keywords:
                    multi_reps.append(f"关键词: {', '.join(keywords)}")
                
                # 摘要表示
                summary = self._generate_summary(chunk)
                if summary:
                    multi_reps.append(f"摘要: {summary}")
            
            return multi_reps
            
        except Exception as e:
            logger.error(f"创建多表示失败: {e}")
            return [text]  # 失败时返回原始文本
    
    def create_raptor_index(self, texts: list, persist_directory: str = "./chroma_db"):
        """
        创建Raptor索引（分层摘要+聚类）
        
        :param texts: 要索引的文本列表
        :param persist_directory: 向量存储的持久化目录
        :return: 创建的向量存储
        """
        try:
            logger.info(f"开始创建Raptor索引，文本数量: {len(texts)}")
            
            # 1. 生成文本嵌入
            # 这里和langchain好像不兼容，所以用openai的api直接调用
            resp = self.embeddings.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            embeddings = [np.array(item.embedding) for item in resp.data]
            logger.info(f"文本嵌入生成完成")
            
            # 2. 聚类 - 避免聚类数量为0
            num_clusters = max(1, min(5, len(texts) // 2))
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings)
            logger.info(f"聚类完成，聚类数量: {num_clusters}")
            
            # 3. 为每个聚类生成总结
            cluster_summaries = []
            for i in range(num_clusters):
                # 获取该聚类的所有文本
                cluster_texts = [texts[j] for j, c in enumerate(clusters) if c == i]
                
                # 生成聚类总结
                if len(cluster_texts) == 1:
                    # 单个文本直接使用
                    summary = cluster_texts[0]
                else:
                    # 多个文本生成总结
                    combined_text = "\n".join(cluster_texts)
                    summary = self._generate_summary(combined_text)
                    if not summary:
                        # 总结失败时使用第一个文本
                        summary = cluster_texts[0]
                
                cluster_summaries.append(summary)
            
            logger.info(f"聚类总结生成完成")
            # 4. 对聚类总结embedding
            resp = self.embeddings.embeddings.create(
                model=self.embedding_model,
                input=cluster_summaries
            )
            cluster_embeddings = [np.array(item.embedding) for item in resp.data]
            logger.info(f"聚类总结嵌入生成完成")
            
            # 5. 创建向量存储
            chromadb_collection = self.save_embeddings(cluster_summaries, cluster_embeddings, persist_directory)
            
            logger.info(f"Raptor索引创建完成")
            return chromadb_collection
            
        except Exception as e:
            logger.error(f"创建Raptor索引失败: {e}，使用简单索引")
            # 失败时创建简单的向量存储
            try:
                simple_vectorstore = self.save_embeddings(texts, embeddings, persist_directory)
                return simple_vectorstore
            except Exception as e:
                logger.error(f"简单索引也创建失败: {e}，无法创建索引")
                raise e
    
    def _extract_keywords(self, text: str) -> list:
        """
        提取文本的关键词
        
        :param text: 输入文本
        :return: 关键词列表
        """
        # 简单的关键词提取实现
        # 实际应用中可以替换为更复杂的关键词提取算法
        import jieba.analyse
        try:
            keywords = jieba.analyse.extract_tags(text, topK=5)
            return keywords
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return []
    
    def _generate_summary(self, text: str) -> str:
        """
        生成文本摘要
        
        :param text: 输入文本
        :return: 文本摘要
        """
        # 使用简单的截断作为替代
        # 实际应用中可以调用LLM生成摘要
        try:
            max_length = 200
            if len(text) <= max_length:
                return text
            else:
                return text[:max_length] + "..."
        except Exception as e:
            logger.error(f"摘要生成失败: {e}")
            return ""
        
    def read_index(self, persist_directory: str = "./chroma_db"):
        """
        读取已存在的Raptor索引
        
        :param persist_directory: 向量存储的持久化目录
        :return: 读取的向量存储
        """
        try:
            chromadb_collection = chromadb.PersistentClient(path=persist_directory).get_or_create_collection(name="default")
            logger.info(f"Raptor索引读取完成")
            return chromadb_collection
        except Exception as e:
            logger.error(f"读取Raptor索引失败: {e}")
            return None
    def save_embeddings(self, chunks: list, embeddings: list, persist_directory: str = "./chroma_db"):
        chromadb_client = chromadb.PersistentClient(path=persist_directory)
        chromadb_collection = chromadb_client.get_or_create_collection(name="default")
        chromadb_collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"str{i}" for i in range(len(chunks))]
        )
        return chromadb_collection


# Multi-Query Prompt - 生成多个相关查询
multi_query_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个查询扩展专家，需要将用户查询扩展为多个相关查询。"),
    ("system", "要求："
    "1. 生成3-5个相关查询"
    "2. 保持查询意图不变"
    "3. 使用不同的表述方式"
    "4. 每个查询用换行分隔"
    "5. 不要添加任何解释或额外内容"),
    ("human", "原始查询：{query}"),
])

# HyDE Prompt - 生成假设回答用于检索
hyde_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个知识渊博的专家，需要根据用户查询生成一个假设的详细回答。"),
    ("system", "要求："
    "1. 假设你拥有所有必要的知识"
    "2. 生成一个详细、具体的回答"
    "3. 使用自然语言，不要使用列表或特殊格式"
    "4. 不要添加任何解释或额外内容"),
    ("human", "查询：{query}"),
])

# Query Decomposition Prompt - 分解复杂查询
query_decomposition_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个查询分解专家，需要将复杂查询分解为多个简单的子查询。"),
    ("system", "要求："
    "1. 识别查询中的多个子问题"
    "2. 每个子问题用换行分隔"
    "3. 保持子问题的独立性"
    "4. 不要添加任何解释或额外内容"),
    ("human", "复杂查询：{query}"),
])

def get_query_transformer(model_name: str = "gpt-4o-mini-2024-07-18", base_url: str = "https://free.v36.cm/v1"):
    """
    获取Query Transformer组件，包含多种查询转换功能
    
    :param model_name: 使用的LLM模型名称
    :param base_url: LLM API的基础URL
    :return: 包含多种查询转换功能的字典
    """
    transformer_model = ChatOpenAI(
        model_name=model_name,
        temperature=0.7,
        max_tokens=500,
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    return {
        "multi_query": multi_query_prompt | transformer_model,
        "hyde": hyde_prompt | transformer_model,
        "decompose": query_decomposition_prompt | transformer_model
    }