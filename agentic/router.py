from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

router_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个智能路由决策器，需要决定如何处理用户的查询请求。"),
    ("system", "判断标准："
    "1. 如果查询可以通过常识或简单推理直接回答，选择'direct'"
    "2. 如果查询需要外部知识或具体信息支持，选择'retrieve'"
    "3. 如果查询复杂需要分解为多个子问题，选择'decompose'"),
    ("system", "注意：请只输出'direct'、'retrieve'或'decompose'中的一个，不要添加任何其他内容。"),
    ("human", "查询：{query}"),
])

def get_agentic_router(model_name: str = "gpt-4o-mini-2024-07-18", base_url: str = "https://free.v36.cm/v1"):
    """
    获取Agentic Router组件
    
    :param model_name: 使用的LLM模型名称
    :param base_url: LLM API的基础URL
    :return: 路由决策链
    """
    router_model = ChatOpenAI(
        model_name=model_name,
        temperature=0.3,
        max_tokens=100,
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    router_chain = router_prompt | router_model
    return router_chain