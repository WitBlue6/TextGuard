from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

load_dotenv()

from .prompt import GRAMMAR_CHECK_PROMPT, ENTITY_EXTRACT_PROMPT, ENTITY_CONSISTENCY_CHECK_PROMPT, MEMORY_SUMMARY_PROMPT, CONSISTENCY_CORRECT_PROMPT, FEEDBACK_SUMMARY_PROMPT
from .memory import SimpleMemory

memory_store = {}

def get_memory(session_id: str) -> SimpleMemory:
    if session_id not in memory_store:
        memory_store[session_id] = SimpleMemory()
    return memory_store[session_id]

def get_grammar_check_chain(model_name: str ="gpt-4o-mini-2024-07-18", base_url: str ="https://free.v36.cm/v1"):
    grammar_check_prompt = ChatPromptTemplate.from_messages([
        ("system", GRAMMAR_CHECK_PROMPT),
        ("human", "{new_message}"),
    ])
    grammar_check_model = ChatOpenAI(
        model_name=model_name,
        temperature=0.7,
        max_tokens=1024,
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    grammar_check_chain = grammar_check_prompt | grammar_check_model
    return grammar_check_chain

def get_grammar_check_chain_with_memory(model_name: str ="gpt-4o-mini-2024-07-18", base_url: str ="https://free.v36.cm/v1"):
    grammar_check_prompt_with_memory = ChatPromptTemplate.from_messages([
        ("system", GRAMMAR_CHECK_PROMPT),
        ("system", "当前对话历史：\n{history}"),
        ("human", "{new_message}"),
    ])
    grammar_check_model = ChatOpenAI(
        model_name=model_name,
        temperature=0.7,
        max_tokens=1024,
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    grammar_check_with_memory = RunnableWithMessageHistory(
        grammar_check_prompt_with_memory | grammar_check_model,
        get_memory,
        input_messages_key="new_message", 
        history_messages_key="history",    # memory 要注入的位置
    )
    return grammar_check_with_memory

def get_entity_extract_chain(model_name: str ="gpt-4o-mini-2024-07-18", base_url: str ="https://free.v36.cm/v1"):
    entity_extract_prompt = ChatPromptTemplate.from_messages([
        ("system", ENTITY_EXTRACT_PROMPT),
        ("system", "当前对话历史:\n{history}"),
        ("human", "{new_message}"),
    ])
    entity_extract_model = ChatOpenAI(
        model_name=model_name,
        temperature=0.7,
        max_tokens=1024,
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    entity_extract_chain = RunnableWithMessageHistory(
        entity_extract_prompt | entity_extract_model,
        get_memory,
        input_messages_key="new_message", 
        history_messages_key="history",    # memory 要注入的位置
    )
    return entity_extract_chain

def get_entity_consistency_check_chain(model_name: str ="gpt-4o-mini-2024-07-18", base_url: str ="https://free.v36.cm/v1"):
    entity_consistency_check_prompt = ChatPromptTemplate.from_messages([
        ("system", ENTITY_CONSISTENCY_CHECK_PROMPT),
        ("human", "{new_message}"),
    ])
    entity_consistency_check_model = ChatOpenAI(
        model_name=model_name,
        temperature=0.7,
        max_tokens=1024,
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    entity_consistency_check_chain = entity_consistency_check_prompt | entity_consistency_check_model
    return entity_consistency_check_chain

def get_memory_summary_chain(model_name: str ="gpt-4o-mini-2024-07-18", base_url: str ="https://free.v36.cm/v1"):
    memory_summary_prompt = ChatPromptTemplate.from_messages([
        ("system", MEMORY_SUMMARY_PROMPT),
        ("human", "{new_message}"),
    ])
    memory_summary_model = ChatOpenAI(
        model_name=model_name,
        temperature=0.7,
        max_tokens=1024,
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    memory_summary_chain = memory_summary_prompt | memory_summary_model
    return memory_summary_chain

def get_consistency_correct_chain(model_name: str ="gpt-4o-mini-2024-07-18", base_url: str ="https://free.v36.cm/v1"):
    consistency_correct_prompt = ChatPromptTemplate.from_messages([
        ("system", CONSISTENCY_CORRECT_PROMPT),
        ("human", "{new_message}"),
    ])
    consistency_correct_model = ChatOpenAI(
        model_name=model_name,
        temperature=0.7,
        max_tokens=1024,
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    consistency_correct_chain = consistency_correct_prompt | consistency_correct_model
    return consistency_correct_chain

# 在文件末尾添加以下内容
def get_feedback_summary_chain(model_name: str ="gpt-4o-mini-2024-07-18", base_url: str ="https://free.v36.cm/v1"):
    
    feedback_summary_prompt = ChatPromptTemplate.from_messages([
        ("system", FEEDBACK_SUMMARY_PROMPT),
        ("human", "语法检查结果：{grammar_results}\n用户反馈：{user_feedback}"),
    ])
    feedback_summary_model = ChatOpenAI(
        model_name=model_name,
        temperature=0.7,
        max_tokens=512,
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    feedback_summary_chain = feedback_summary_prompt | feedback_summary_model
    return feedback_summary_chain