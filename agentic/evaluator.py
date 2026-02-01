from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
import logging

logger = logging.getLogger(__name__)

class SelfEvaluator:
    def __init__(self, model_name: str = "gpt-4o-mini-2024-07-18", base_url: str = "https://free.v36.cm/v1"):
        """
        初始化自我评估器
        
        :param model_name: 使用的LLM模型名称
        :param base_url: LLM API的基础URL
        """
        self.model = ChatOpenAI(
            model_name=model_name,
            temperature=0.3,  # 低温度保证评估的客观性
            max_tokens=1000,
            base_url=base_url,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    
    def evaluate_retrieval(self, query: str, docs: list):
        """
        评估检索结果的质量
        
        :param query: 用户查询
        :param docs: 检索到的文档列表
        :return: 评估结果，包含相关性、完整性等维度
        """
        try:
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
            docs_str = "\n".join([f"文档 {i+1}: {doc.page_content[:300]}..." for i, doc in enumerate(docs)])
            
            result = self.model.invoke(eval_prompt.format(query=query, docs=docs_str)).content.strip()
            logger.debug(f"检索结果评估: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"评估检索结果失败: {e}")
            return "{\"error\": \"评估失败\"}"
    
    def evaluate_answer(self, query: str, answer: str, docs: list):
        """
        评估生成的回答质量
        
        :param query: 用户查询
        :param answer: 生成的回答
        :param docs: 用于生成回答的文档
        :return: 评估结果，包含准确性、相关性、完整性等维度
        """
        try:
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
            docs_str = "\n".join([f"文档 {i+1}: {doc.page_content[:300]}..." for i, doc in enumerate(docs)])
            
            result = self.model.invoke(eval_prompt.format(query=query, docs=docs_str, answer=answer)).content.strip()
            logger.debug(f"回答质量评估: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"评估回答质量失败: {e}")
            return "{\"error\": \"评估失败\"}"
    
    def evaluate_consistency_correction(self, original_text: str, corrected_text: str, consistency_issues: list):
        """
        评估一致性修正的质量
        
        :param original_text: 原始文本
        :param corrected_text: 修正后的文本
        :param consistency_issues: 原始一致性问题
        :return: 评估结果
        """
        try:
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
            )).content.strip()
            
            logger.debug(f"一致性修正评估: {result}")
            return result
            
        except Exception as e:
            logger.error(f"评估一致性修正失败: {e}")
            return "{\"error\": \"评估失败\"}"