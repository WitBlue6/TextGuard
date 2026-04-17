"""
本地路由决策器
支持使用本地LLM进行智能路由决策
"""
from langchain_core.prompts import ChatPromptTemplate
from llm.model_local import get_local_model
import logging

logger = logging.getLogger(__name__)


def get_agentic_router_local(model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                             model_path: str = None,
                             device: str = None,
                             quantization: bool = False):
    """
    获取本地Agentic Router组件

    :param model_name: 使用的LLM模型名称
    :param model_path: 本地模型路径
    :param device: 设备
    :param quantization: 是否量化
    :return: 路由决策链
    """
    try:
        # 获取本地模型
        hf_model = get_local_model(model_name, model_path, device, quantization)

        # 获取prompt
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个智能路由决策器，需要决定如何处理用户的查询请求。"),
            ("system", "判断标准："
            "1. 如果查询可以通过你的已有知识或简单推理直接回答，选择'direct'，如'什么是Python？'、'地球离太阳有多远？'等"
            "2. 如果查询需要外部知识或具体信息支持，选择'retrieve'，如'2023年北京天气如何？'、'最新的AI研究动态？'等"
            "3. 如果查询复杂需要分解为多个子问题，选择'decompose'，如'比较不同排序算法的性能'、'如何解释量子计算在当前技术中的作用'等"),
            ("system", "注意：请只输出'direct'、'retrieve'或'decompose'中的一个，不要添加任何其他内容。"),
            ("human", "查询：{query}"),
        ])

        # 创建路由链
        router_chain = router_prompt | hf_model

        logger.info("本地Agentic Router组件初始化完成")
        return router_chain

    except Exception as e:
        logger.error(f"初始化本地Agentic Router失败: {e}")
        # 失败时返回默认直接模式
        return None
