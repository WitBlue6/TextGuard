"""
本地自我评估器
支持使用本地LLM进行质量评估
"""
from langchain_core.prompts import ChatPromptTemplate
from llm.model_local import get_local_model
import logging

logger = logging.getLogger(__name__)


class LocalSelfEvaluator:
    """
    本地自我评估器
    支持使用本地LLM进行检索结果和回答质量的评估
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 model_path: str = None,
                 device: str = None,
                 quantization: bool = False):
        """
        初始化本地自我评估器

        :param model_name: 使用的LLM模型名称
        :param model_path: 本地模型路径
        :param device: 设备
        :param quantization: 是否量化
        """
        try:
            # 获取本地模型
            self.model = get_local_model(model_name, model_path, device, quantization)

            logger.info("本地自我评估器初始化完成")
        except Exception as e:
            logger.error(f"初始化本地自我评估器失败: {e}")
            self.model = None

    def evaluate_retrieval_local(self, query: str, docs: list) -> str:
        """
        评估检索结果的质量（本地版本）

        :param query: 用户查询
        :param docs: 检索到的文档列表
        :return: 评估结果JSON字符串
        """
        try:
            if self.model is None:
                return "{\"error\": \"评估模型未初始化\"}"

            eval_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个检索结果评估专家，需要评估检索结果与查询的相关性和质量。"),
                ("system", "请从以下维度评估："
                "1. 相关性：检索结果与查询的相关程度（1-5分）\n"
                "2. 完整性：检索结果是否包含回答问题所需的所有信息（1-5分）\n"
                "3. 准确性：检索结果中的信息是否准确（1-5分）\n"
                "4. 简洁性：检索结果是否简洁明了，不含冗余信息（1-5分）\n"
                "5. 总体质量：综合以上维度的总体评分（1-5分）\n"
                "6. 改进建议：如何改进检索效果\n"
                ),
                ("system", "请使用JSON格式输出评估结果，包含'relevance'、'completeness'、'accuracy'、'conciseness'、'overall_quality'和'improvement_suggestions'字段。"),
                ("human", "查询：{query}\n\n检索结果：{docs}"),
            ])

            # 构建文档字符串
            docs_str = "\n".join([f"文档 {i+1}: {doc[:1000]}..." for i, doc in enumerate(docs)])

            result = self.model.invoke(eval_prompt.format(query=query, docs=docs_str))
            # 确保result是字符串
            result_str = str(result)
            # 提取Assistant后面的内容
            if "Assistant:" in result_str:
                assistant_index = result_str.find("Assistant:")
                if assistant_index != -1:
                    result_str = result_str[assistant_index + len("Assistant:"):].strip()
            result = result_str.strip()
            logger.debug(f"检索结果评估: {result}")

            return result

        except Exception as e:
            logger.error(f"评估检索结果失败: {e}")
            return "{\"error\": \"评估失败\"}"

    def evaluate_answer_local(self, query: str, answer: str, docs: list) -> str:
        """
        评估生成的回答质量（本地版本）

        :param query: 用户查询
        :param answer: 生成的回答
        :param docs: 用于生成回答的文档
        :return: 评估结果JSON字符串
        """
        try:
            if self.model is None:
                return "{\"error\": \"评估模型未初始化\"}"

            eval_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个回答质量评估专家，需要评估回答的质量。"),
                ("system", "请从以下维度评估："
                "1. 准确性：回答中的信息是否准确（1-5分）\n"
                "2. 相关性：回答与查询的相关程度（1-5分）\n"
                "3. 完整性：回答是否完整覆盖了查询的所有需求（1-5分）\n"
                "4. 清晰度：回答是否清晰易懂（1-5分）\n"
                "5. 一致性：回答是否与检索到的文档一致（1-5分）\n"
                "6. 总体质量：综合以上维度的总体评分（1-5分）\n"
                "7. 改进建议：如何改进回答质量\n"
                ),
                ("system", "请使用JSON格式输出评估结果，包含'accuracy'、'relevance'、'completeness'、'clarity'、'consistency'、'overall_quality'和'improvement_suggestions'字段。"),
                ("human", "查询：{query}\n\n检索到的文档：{docs}\n\n生成的回答：{answer}"),
            ])

            # 构建文档字符串
            docs_str = "\n".join([f"文档 {i+1}: {doc[:1000]}..." for i, doc in enumerate(docs)])

            result = self.model.invoke(eval_prompt.format(query=query, docs=docs_str, answer=answer))
            # 确保result是字符串
            result_str = str(result)
            # 提取Assistant后面的内容
            if "Assistant:" in result_str:
                assistant_index = result_str.find("Assistant:")
                if assistant_index != -1:
                    result_str = result_str[assistant_index + len("Assistant:"):].strip()
            result = result_str.strip()
            logger.debug(f"回答质量评估: {result}")

            return result

        except Exception as e:
            logger.error(f"评估回答质量失败: {e}")
            return "{\"error\": \"评估失败\"}"

    def evaluate_consistency_correction_local(self, original_text: str, corrected_text: str, consistency_issues: list) -> str:
        """
        评估一致性修正的质量（本地版本）

        :param original_text: 原始文本
        :param corrected_text: 修正后的文本
        :param consistency_issues: 原始一致性问题
        :return: 评估结果JSON字符串
        """
        try:
            if self.model is None:
                return "{\"error\": \"评估模型未初始化\"}"

            eval_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个文本一致性修正评估专家，需要评估文本一致性修正的质量。"),
                ("system", "请从以下维度评估："
                "1. 问题解决：是否成功解决了所有一致性问题（1-5分）\n"
                "2. 保留原意：是否保留了原始文本的意思（1-5分）\n"
                "3. 自然流畅：修正后的文本是否自然流畅（1-5分）\n"
                "4. 准确性：修正后的文本是否准确（1-5分）\n"
                "5. 总体质量：综合以上维度的总体评分（1-5分）\n"
                "6. 改进建议：如何改进修正质量\n"
                ),
                ("system", "请使用JSON格式输出评估结果，包含'problem_resolution'、'original_meaning_preserved'、'naturalness'、'accuracy'、'overall_quality'和'improvement_suggestions'字段。"),
                ("human", "原始文本：{original_text}\n\n原始一致性问题：{consistency_issues}\n\n修正后的文本：{corrected_text}"),
            ])

            result = self.model.invoke(eval_prompt.format(
                original_text=original_text[:500] + ("..." if len(original_text) > 500 else ""),
                consistency_issues=str(consistency_issues),
                corrected_text=corrected_text[:500] + ("..." if len(corrected_text) > 500 else "")
            ))
            # 确保result是字符串
            result_str = str(result)
            # 提取Assistant后面的内容
            if "Assistant:" in result_str:
                assistant_index = result_str.find("Assistant:")
                if assistant_index != -1:
                    result_str = result_str[assistant_index + len("Assistant:"):].strip()
            result = result_str.strip()

            logger.debug(f"一致性修正评估: {result}")
            return result

        except Exception as e:
            logger.error(f"评估一致性修正失败: {e}")
            return "{\"error\": \"评估失败\"}"

    def evaluate_query_local(self, query: str) -> str:
        """
        评估查询的质量（本地版本）

        :param query: 用户查询
        :return: 评估结果JSON字符串
        """
        try:
            if self.model is None:
                return "{\"error\": \"评估模型未初始化\"}"

            eval_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个查询质量评估专家，需要评估查询的清晰度和可回答性。"),
                ("system", "请从以下维度评估："
                "1. 清晰度：查询是否清晰明确（1-5分）\n"
                "2. 具体性：查询是否具体，不是过于泛化（1-5分）\n"
                "3. 完整性：查询是否包含了所有必要的信息（1-5分）\n"
                "4. 可回答性：这个查询是否容易被回答（1-5分）\n"
                "5. 总体质量：综合以上维度的总体评分（1-5分）\n"
                "6. 改进建议：如何改进查询质量\n"
                ),
                ("system", "请使用JSON格式输出评估结果，包含'clarity'、'specificity'、'completeness'、'answerability'、'overall_quality'和'improvement_suggestions'字段。"),
                ("human", "查询：{query}"),
            ])

            result = self.model.invoke(eval_prompt.format(query=query))
            # 确保result是字符串
            result_str = str(result)
            # 提取Assistant后面的内容
            if "Assistant:" in result_str:
                assistant_index = result_str.find("Assistant:")
                if assistant_index != -1:
                    result_str = result_str[assistant_index + len("Assistant:"):].strip()
            result = result_str.strip()
            logger.debug(f"查询质量评估: {result}")

            return result

        except Exception as e:
            logger.error(f"评估查询质量失败: {e}")
            return "{\"error\": \"评估失败\"}"

    def evaluate_entity_consistency_local(self, entity: dict, consistency_result: dict) -> str:
        """
        评估实体一致性检查结果（本地版本）

        :param entity: 实体信息
        :param consistency_result: 一致性检查结果
        :return: 评估结果JSON字符串
        """
        try:
            if self.model is None:
                return "{\"error\": \"评估模型未初始化\"}"

            eval_prompt = ChatPromptTemplate.from_messages([
                ("system", "你是一个实体一致性评估专家，需要评估一致性检查结果的合理性。"),
                ("system", "请评估以下一致性检查结果："),
                ("human", "实体信息：{entity}\n\n一致性检查结果：{consistency_result}\n\n请评估：\n1. 评估结果是否合理\n2. 检查是否遗漏了重要的一致性问题\n3. 改进建议：如何改进一致性检查\n"),
            ])

            result = self.model.invoke(eval_prompt.format(
                entity=entity,
                consistency_result=consistency_result
            ))
            # 确保result是字符串
            result_str = str(result)
            # 提取Assistant后面的内容
            if "Assistant:" in result_str:
                assistant_index = result_str.find("Assistant:")
                if assistant_index != -1:
                    result_str = result_str[assistant_index + len("Assistant:"):].strip()
            result = result_str.strip()

            logger.debug(f"实体一致性评估: {result}")
            return result

        except Exception as e:
            logger.error(f"评估实体一致性失败: {e}")
            return "{\"error\": \"评估失败\"}"