"""
本地实体管理模块
支持从本地LLM提取实体
"""
import json
import uuid
import logging
from typing import Dict, Any, List, Optional

from .model_local import get_entity_extract_chain_local
from .entity import UIEntity, EntityStore
from .prompt import ENTITY_EXTRACT_PROMPT

logger = logging.getLogger(__name__)


def extract_entities_local(chain, text: str,
                           model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                           model_path: str = None,
                           device: str = None,
                           quantization: bool = False) -> List[UIEntity]:
    """
    从文本中提取实体（本地LLM版本）

    :param chain: 实体提取链（从model_local.py获取）
    :param text: 输入的文本
    :param model_name: 模型名称
    :param model_path: 本地模型路径
    :param device: 设备
    :param quantization: 是否量化
    :return: 提取到的实体列表
    """
    try:
        # 如果没有提供chain，创建一个新的
        if chain is None:
            chain = get_entity_extract_chain_local(
                model_name=model_name,
                model_path=model_path,
                device=device,
                quantization=quantization
            )

        logger.info(f"使用本地模型提取实体，文本长度: {len(text)}")

        # 调用chain提取实体
        result = chain.invoke({"new_message": text},
                              config={"session_id": "local_entity_extraction"}
                              ).content

        logger.debug(f"原始LLM响应: {result}")

        # 清理响应文本
        if "```" in result:
            result = result.split("```")[-1].strip()

        # 提取JSON代码块
        import re
        json_match = re.search(r'```json\s*\n(.*?)\n```', result, re.DOTALL)
        if json_match:
            result = json_match.group(1).strip()

        logger.debug(f"提取的JSON实体: {result}")

        # 解析JSON
        raw_entities = json.loads(result)

        # 转化为 UIEntity 列表
        entities = [UIEntity(entity_id=str(uuid.uuid4()), **entity) for entity in raw_entities]

        logger.info(f"成功提取 {len(entities)} 个实体")
        return entities

    except Exception as e:
        logger.error(f"本地实体提取失败: {e}")
        logger.debug("LLM响应详情:", exc_info=True)
        return []


def check_entity_consistency_local(chain, entity: UIEntity,
                                    enhanced_input: Optional[str] = None,
                                    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                                    model_path: str = None,
                                    device: str = None,
                                    quantization: bool = False) -> Dict[str, Any]:
    """
    检查实体的一致性（本地LLM版本）

    :param chain: 实体一致性检查链
    :param entity: 待检查的实体
    :param enhanced_input: 增强的输入信息，如检索到的相关内容
    :param model_name: 模型名称
    :param model_path: 本地模型路径
    :param device: 设备
    :param quantization: 是否量化
    :return: 检查结果
    """
    try:
        entity_json = entity.model_dump_json()

        # 如果有增强输入，则合并
        if enhanced_input:
            input_text = f"{enhanced_input}\n\n实体信息: {entity_json}"
        else:
            input_text = entity_json

        logger.info(f"使用本地模型检查实体一致性: {entity.name}")

        # 调用chain检查一致性
        result = chain.invoke({"new_message": input_text},
                              config={"session_id": "local_consistency_check"}
                              ).content

        logger.debug(f"原始LLM响应: {result}")

        # 清理响应文本
        if "```" in result:
            result = result.split("```")[-1].strip()

        # 提取JSON代码块
        import re
        json_match = re.search(r'```json\s*\n(.*?)\n```', result, re.DOTALL)
        if json_match:
            result = json_match.group(1).strip()

        logger.debug(f"一致性检查JSON结果: {result}")

        return json.loads(result)

    except Exception as e:
        logger.error(f"本地实体一致性检查失败: {e}")
        logger.debug("LLM响应详情:", exc_info=True)
        return {
            "entity_name": entity.name,
            "has_conflict": False,
            "conflicts": [],
            "explanation": f"检查失败: {str(e)}"
        }


def summarize_entity_memory_local(chain, chunk: str,
                                   model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                                   model_path: str = None,
                                   device: str = None,
                                   quantization: bool = False) -> str:
    """
    对输入chunking后的实体列表进行总结（本地LLM版本）

    :param chain: 实体内存总结链
    :param chunk: 输入的文本chunk
    :param model_name: 模型名称
    :param model_path: 本地模型路径
    :param device: 设备
    :param quantization: 是否量化
    :return: 总结文本
    """
    try:
        # 如果没有提供chain，创建一个新的
        if chain is None:
            chain = get_memory_summary_chain_local(
                model_name=model_name,
                model_path=model_path,
                device=device,
                quantization=quantization
            )

        logger.info(f"使用本地模型总结实体内存，chunk长度: {len(chunk)}")

        # 调用chain总结
        result = chain.invoke({"new_message": chunk},
                              config={"session_id": "local_memory_summary"}
                              ).content

        logger.debug(f"原始LLM响应: {result}")

        # 清理响应文本
        if "```" in result:
            result = result.split("```")[-1].strip()

        logger.debug(f"总结结果: {result}")

        return result

    except Exception as e:
        logger.error(f"本地实体内存总结失败: {e}")
        logger.debug("LLM响应详情:", exc_info=True)
        return chunk  # 失败时返回原chunk


def check_entity_consistency_with_enhancement_local(
        chain, entity: UIEntity,
        retrieval_results: List[str],
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        model_path: str = None,
        device: str = None,
        quantization: bool = False) -> Dict[str, Any]:
    """
    使用RAG增强检索结果进行实体一致性检查（本地LLM版本）

    :param chain: 实体一致性检查链
    :param entity: 待检查的实体
    :param retrieval_results: 检索到的相关文档列表
    :param model_name: 模型名称
    :param model_path: 本地模型路径
    :param device: 设备
    :param quantization: 是否量化
    :return: 检查结果
    """
    try:
        # 合并检索结果
        enhanced_context = "\n\n".join(retrieval_results)
        entity_json = entity.model_dump_json()

        # 构建增强的输入
        enhanced_input = f"实体信息: {entity_json}\n\n检索到的相关信息: {enhanced_context}\n\n请检查该实体的一致性"

        logger.info(f"使用RAG增强检查实体: {entity.name}，检索到 {len(retrieval_results)} 个文档")

        # 调用chain检查一致性
        result = chain.invoke({"new_message": enhanced_input},
                              config={"session_id": "local_rag_consistency_check"}
                              ).content

        logger.debug(f"原始LLM响应: {result}")

        # 清理响应文本
        if "```" in result:
            result = result.split("```")[-1].strip()

        # 提取JSON代码块
        import re
        json_match = re.search(r'```json\s*\n(.*?)\n```', result, re.DOTALL)
        if json_match:
            result = json_match.group(1).strip()

        logger.debug(f"RAG增强的一致性检查JSON结果: {result}")

        return json.loads(result)

    except Exception as e:
        logger.error(f"本地RAG增强实体一致性检查失败: {e}")
        logger.debug("LLM响应详情:", exc_info=True)
        # 失败时直接检查，不使用RAG增强
        return check_entity_consistency_local(
            chain, entity,
            model_name=model_name,
            model_path=model_path,
            device=device,
            quantization=quantization
        )
