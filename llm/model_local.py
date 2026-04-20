"""
本地LLM模型调用模块
支持HuggingFace模型和本地模型推理
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_huggingface import HuggingFacePipeline
import os
from dotenv import load_dotenv
import logging
from typing import Optional

load_dotenv()

logger = logging.getLogger(__name__)

# 全局模型缓存
_model_cache = {}
_device = None

def get_local_device():
    """获取本地设备"""
    global _device
    if _device is None:
        try:
            # 尝试检查CUDA是否可用
            if torch.cuda.is_available():
                # 进一步检查CUDA设备是否可以使用
                torch.cuda.get_device_name(0)
                device = "cuda"
            else:
                device = "cpu"
        except Exception as e:
            # 如果CUDA检查失败，回退到CPU
            logger.warning(f"CUDA检查失败: {e}，回退到CPU")
            device = "cpu"
        _device = device
        logger.info(f"使用设备: {device}")
    return _device

def get_local_model(model_name: str, model_path: Optional[str] = None,
                    device: Optional[str] = None,
                    quantization: bool = False,
                    gpus: Optional[list] = None):
    """
    获取本地LLM模型

    :param model_name: HuggingFace模型名称，如 'Qwen/Qwen2.5-7B-Instruct'
    :param model_path: 本地模型路径，优先于model_name
    :param device: 设备 ('cuda', 'cpu', 'mps')
    :param quantization: 是否使用量化模型
    :return: 本地LLM模型实例
    """
    global _model_cache

    cache_key = model_path or model_name

    if cache_key in _model_cache:
        logger.info(f"从缓存加载模型: {cache_key}")
        return _model_cache[cache_key]

    try:
        if model_path:
            model = model_path
        else:
            model = model_name

        logger.info(f"加载本地模型: {model}")
        logger.info(f"设备: {device or get_local_device()}, 量化: {quantization}")

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 加载模型
        if gpus:
            logger.info(f"指定GPU: {gpus}")
            if isinstance(gpus, str):
                os.environ["CUDA_VISIBLE_DEVICES"] = gpus
                # gpus:"0,1,2,3", 转为list
                gpus = gpus.split(",")
            elif isinstance(gpus, list):
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
            else:
                logger.error(f"Invalid gpus format: {gpus}, Support Type: str or list")
            load_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto"
            }
        else:
            # 检查CUDA是否可用
            if torch.cuda.is_available():
                logger.info("CUDA可用")
                load_kwargs = {
                    "trust_remote_code": True,
                    "device_map": "auto"
                }
            else:
                device = "cpu" if device == "cuda" else device
                logger.info(f"CUDA不可用, 使用: {device or get_local_device()}")

                load_kwargs = {
                    "trust_remote_code": True,
                    "device_map": device or get_local_device()
                }

        if quantization:
            load_kwargs["quantization_config"] = {
                "load_in_8bit": True,
                "llm_int8_threshold": 6.0,
            }
            load_kwargs["device_map"] = "auto"
        
        # 加载模型
        model_obj = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
        # 使用compile优化加速
        # model_obj = torch.compile(model_obj, mode="max-autotune", dynamic=True)

        # 创建生成pipeline
        pipe = pipeline(
            "text-generation",
            model=model_obj,
            tokenizer=tokenizer,
            device_map=load_kwargs["device_map"],
            max_new_tokens=1024,
            max_length=None,  # 显式设置max_length=None，避免与max_new_tokens冲突
            temperature=0.0,
            top_p=1,
            do_sample=False,
            batch_size=32,
        )

        # 包装为LangChain可用的HuggingFacePipeline
        hf_pipeline = HuggingFacePipeline(
            pipeline=pipe,
            #model_kwargs={"temperature": 0.7, "max_new_tokens": 1024}
        )

        # 缓存模型
        _model_cache[cache_key] = hf_pipeline

        logger.info(f"模型加载完成: {cache_key}")
        return hf_pipeline

    except Exception as e:
        logger.error(f"加载本地模型失败: {e}")
        raise

def cleanup_model_cache():
    """清理模型缓存"""
    global _model_cache
    for key, model in _model_cache.items():
        if hasattr(model, 'pipeline'):
            del model.pipeline
    _model_cache.clear()
    logger.info("模型缓存已清理")

# 消息历史内存
memory_store = {}

def get_local_memory(session_id: str):
    """获取本地内存"""
    if session_id not in memory_store:
        memory_store[session_id] = []
    return memory_store[session_id]

def clear_local_memory(session_id: str):
    """清理本地内存"""
    if session_id in memory_store:
        memory_store[session_id] = []

def get_grammar_check_chain_local(model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                                   model_path: str = None,
                                   device: str = None,
                                   quantization: bool = False,
                                   gpus: Optional[list] = None):
    """
    获取本地语法检查Chain

    :param model_name: HuggingFace模型名称
    :param model_path: 本地模型路径
    :param device: 设备
    :param quantization: 是否量化
    :return: 语法检查Chain
    """
    from .memory_local import LocalSimpleMemory

    # 获取本地模型
    hf_model = get_local_model(model_name, model_path, device, quantization, gpus)

    # 获取prompt
    from .prompt import GRAMMAR_CHECK_PROMPT

    grammar_check_prompt = ChatPromptTemplate.from_messages([
        ("system", GRAMMAR_CHECK_PROMPT),
        ("human", "{new_message}"),
    ])

    # 创建链
    grammar_check_chain = grammar_check_prompt | hf_model

    return grammar_check_chain

def get_grammar_check_chain_with_memory_local(model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                                               model_path: str = None,
                                               device: str = None,
                                               quantization: bool = False,
                                               gpus: Optional[list] = None):
    """
    获取带记忆的本地语法检查Chain

    :param model_name: HuggingFace模型名称
    :param model_path: 本地模型路径
    :param device: 设备
    :param quantization: 是否量化
    :return: 语法检查Chain
    """
    # 获取本地模型
    hf_model = get_local_model(model_name, model_path, device, quantization, gpus)

    # 获取prompt
    from .prompt import GRAMMAR_CHECK_PROMPT

    grammar_check_prompt = ChatPromptTemplate.from_messages([
        ("system", GRAMMAR_CHECK_PROMPT),
        ("system", "当前对话历史：\n{history}"),
        ("human", "{new_message}"),
    ])

    # 创建带记忆的链
    grammar_check_chain = RunnableWithMessageHistory(
        grammar_check_prompt | hf_model,
        get_local_memory,
        input_messages_key="new_message",
        history_messages_key="history",
    )

    return grammar_check_chain

def get_entity_extract_chain_local(model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                                    model_path: str = None,
                                    device: str = None,
                                    quantization: bool = False,
                                    gpus: Optional[list] = None):
    """
    获取本地实体提取Chain

    :param model_name: HuggingFace模型名称
    :param model_path: 本地模型路径
    :param device: 设备
    :param quantization: 是否量化
    :return: 实体提取Chain
    """
    # 获取本地模型
    hf_model = get_local_model(model_name, model_path, device, quantization, gpus)

    # 获取prompt
    from .prompt import ENTITY_EXTRACT_PROMPT

    entity_extract_prompt = ChatPromptTemplate.from_messages([
        ("system", ENTITY_EXTRACT_PROMPT),
        ("system", "当前对话历史:\n{history}"),
        ("human", "{new_message}"),
    ])

    # 创建带记忆的链
    entity_extract_chain = RunnableWithMessageHistory(
        entity_extract_prompt | hf_model,
        get_local_memory,
        input_messages_key="new_message",
        history_messages_key="history",
    )

    return entity_extract_chain

def get_entity_consistency_check_chain_local(model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                                              model_path: str = None,
                                              device: str = None,
                                              quantization: bool = False,
                                              gpus: Optional[list] = None):
    """
    获取本地实体一致性检查Chain

    :param model_name: HuggingFace模型名称
    :param model_path: 本地模型路径
    :param device: 设备
    :param quantization: 是否量化
    :return: 实体一致性检查Chain
    """
    # 获取本地模型
    hf_model = get_local_model(model_name, model_path, device, quantization, gpus)

    # 获取prompt
    from .prompt import ENTITY_CONSISTENCY_CHECK_PROMPT

    entity_consistency_check_prompt = ChatPromptTemplate.from_messages([
        ("system", ENTITY_CONSISTENCY_CHECK_PROMPT),
        ("human", "{new_message}"),
    ])

    # 创建链
    entity_consistency_check_chain = entity_consistency_check_prompt | hf_model

    return entity_consistency_check_chain

def get_memory_summary_chain_local(model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                                    model_path: str = None,
                                    device: str = None,
                                    quantization: bool = False,
                                    gpus: Optional[list] = None):
    """
    获取本地内存总结Chain

    :param model_name: HuggingFace模型名称
    :param model_path: 本地模型路径
    :param device: 设备
    :param quantization: 是否量化
    :return: 内存总结Chain
    """
    # 获取本地模型
    hf_model = get_local_model(model_name, model_path, device, quantization, gpus)

    # 获取prompt
    from .prompt import MEMORY_SUMMARY_PROMPT

    memory_summary_prompt = ChatPromptTemplate.from_messages([
        ("system", MEMORY_SUMMARY_PROMPT),
        ("human", "{new_message}"),
    ])

    # 创建链
    memory_summary_chain = memory_summary_prompt | hf_model

    return memory_summary_chain

def get_consistency_correct_chain_local(model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                                         model_path: str = None,
                                         device: str = None,
                                         quantization: bool = False,
                                         gpus: Optional[list] = None):
    """
    获取本地一致性修正Chain

    :param model_name: HuggingFace模型名称
    :param model_path: 本地模型路径
    :param device: 设备
    :param quantization: 是否量化
    :return: 一致性修正Chain
    """
    # 获取本地模型
    hf_model = get_local_model(model_name, model_path, device, quantization, gpus)

    # 获取prompt
    from .prompt import CONSISTENCY_CORRECT_PROMPT

    consistency_correct_prompt = ChatPromptTemplate.from_messages([
        ("system", CONSISTENCY_CORRECT_PROMPT),
        ("human", "{new_message}"),
    ])

    # 创建链
    consistency_correct_chain = consistency_correct_prompt | hf_model

    return consistency_correct_chain

def get_feedback_summary_chain_local(model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                                      model_path: str = None,
                                      device: str = None,
                                      quantization: bool = False,
                                      gpus: Optional[list] = None):
    """
    获取本地反馈总结Chain

    :param model_name: HuggingFace模型名称
    :param model_path: 本地模型路径
    :param device: 设备
    :param quantization: 是否量化
    :return: 反馈总结Chain
    """
    # 获取本地模型
    hf_model = get_local_model(model_name, model_path, device, quantization, gpus)

    # 获取prompt
    from .prompt import FEEDBACK_SUMMARY_PROMPT

    feedback_summary_prompt = ChatPromptTemplate.from_messages([
        ("system", FEEDBACK_SUMMARY_PROMPT),
        ("human", "语法检查结果：{grammar_results}\n用户反馈：{user_feedback}"),
    ])

    # 创建链
    feedback_summary_chain = feedback_summary_prompt | hf_model

    return feedback_summary_chain