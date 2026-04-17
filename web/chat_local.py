"""
Web后端本地LLM版本
支持本地HuggingFace模型进行推理
"""
from llm.model_local import get_grammar_check_chain_with_memory_local, get_grammar_check_chain_local, get_entity_extract_chain_local, get_entity_consistency_check_chain_local, get_memory_summary_chain_local, get_consistency_correct_chain_local
from llm.entity_local import extract_entities_local, check_entity_consistency_local, check_entity_consistency_with_enhancement_local, summarize_entity_memory_local
from filereader.reader import chunking, get_text_from_input
from feedback import collect_consistency_feedback, collect_grammar_feedback
import asyncio

# RAG模块导入（本地版本）
from agentic.router_local import get_agentic_router_local
from agentic.indexer_local import LocalAdvancedIndexer
from agentic.retriever_local import LocalIterativeRetriever
from agentic.evaluator_local import LocalSelfEvaluator

import io
from fastapi import APIRouter, UploadFile, File, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import json
import base64
import logging

# 全局缓存日志和结果
TASKS = {}  # task_id -> {"logs": [], "result": None, "done": False}

async def init_rag_components_local(args, rag_mode, log_callback):
    """
    初始化本地RAG组件
    """
    global rag_components_local

    # 检查是否已经初始化
    if 'router' in rag_components_local:
        return

    await log_callback("初始化本地RAG组件...")

    # 初始化RAG组件（本地版本）
    agentic_router = get_agentic_router_local(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        quantization=args.llm_quantization
    )
    advanced_indexer = LocalAdvancedIndexer(
        embedding_model=args.embedding_model,
        model_path=args.model_path,
        device=args.device
    )
    self_evaluator = LocalSelfEvaluator(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        quantization=args.llm_quantization
    )

    # 读取索引
    raptor_index = advanced_indexer.read_index_local(persist_directory=args.chroma_db_dir)
    iterative_retriever = LocalIterativeRetriever(
        vectorstore=raptor_index,
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        embedding_model=args.embedding_model,
        embedding_model_path=args.model_path,
        max_iterations=3,
        k=3
    )

    await log_callback("本地RAG组件初始化完成")

    # 更新记忆中的组件
    rag_components_local['indexer'] = advanced_indexer
    rag_components_local['retriever'] = iterative_retriever
    rag_components_local['router'] = agentic_router
    rag_components_local['evaluator'] = self_evaluator

async def run_consistency_pipeline_local(text: str, args, log_callback, **kwargs):
    """本地版本的一致性检测pipeline"""
    # 初始化本地模型
    grammar_check_with_memory = get_grammar_check_chain_with_memory_local(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        quantization=args.llm_quantization
    )
    entity_extract_chain = get_entity_extract_chain_local(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        quantization=args.llm_quantization
    )
    entity_consistency_check_chain = get_entity_consistency_check_chain_local(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        quantization=args.llm_quantization
    )
    memory_summary_chain = get_memory_summary_chain_local(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        quantization=args.llm_quantization
    )

    logger = kwargs.get("logger", logging.getLogger(__name__))
    cancellation_token = kwargs.get("cancellation_token", None)
    rag_mode = kwargs.get("rag_mode", "none")
    rag_enabled = kwargs.get("rag_enabled", False)

    await log_callback(f"开始运行本地一致性检测pipeline，模型: {args.model_name}")
    logger.info(f"开始运行本地一致性检测pipeline，模型: {args.model_name}")

    # chunking 文本
    await log_callback(f"文本长度: {len(text)}")
    logger.info(f"文本长度: {len(text)}")
    chunks = chunking(text)

    ent_store = {}
    # 处理每个 chunk
    previous_memory = ""
    for i, chunk in enumerate(chunks):
        # 定期检查是否有取消请求
        await asyncio.sleep(0.1)

        # 检查是否需要终止
        if cancellation_token and cancellation_token.is_set():
            await log_callback(f"pipeline已终止", "error")
            logger.info(f"pipeline已终止")
            raise asyncio.CancelledError("Pipeline terminated by user")

        chunk_input = (
            f"前文要点总结:{previous_memory}\n当前输入文本:{chunk}"
            if previous_memory else chunk
        )

        # 使用RAG增强实体提取
        use_rag = rag_mode == "auto" and rag_enabled
        if use_rag:
            ents = await extract_entities_with_rag_local(entity_extract_chain, chunk_input, log_callback, args)
        else:
            ents = extract_entities_local(
                chain=entity_extract_chain,
                text=chunk_input,
                model_name=args.model_name,
                model_path=args.model_path,
                device=args.device,
                quantization=args.llm_quantization
            )

        # 合并实体
        if chunk not in ent_store:
            ent_store[chunk] = ents
        else:
            # 合并实体（简化版）
            for ent in ents:
                if ent not in ent_store[chunk]:
                    ent_store[chunk].append(ent)

        await log_callback(f"第 {i+1} 个 chunk 提取实体: {ents}")
        logger.info(f"第 {i+1} 个 chunk 提取实体: {ents}")

        if i < len(chunks) - 1:
            previous_memory = summarize_entity_memory_local(
                chain=memory_summary_chain,
                chunk=chunk_input,
                model_name=args.model_name,
                model_path=args.model_path,
                device=args.device,
                quantization=args.llm_quantization
            )

    # 检查实体一致性（简化版：只检查第一个chunk的实体）
    await log_callback(f"实体总数: {sum(len(ents) for ents in ent_store.values())}")
    logger.info(f"实体总数: {sum(len(ents) for ents in ent_store.values())}")
    await log_callback(f"开始检查实体一致性")
    logger.info(f"开始检查实体一致性")

    results = []
    # 收集所有实体
    all_entities = []
    for ents in ent_store.values():
        all_entities.extend(ents)

    for ent in all_entities:
        # 定期检查是否有取消请求
        await asyncio.sleep(0.1)

        # 检查是否需要终止
        if cancellation_token and cancellation_token.is_set():
            await log_callback(f"pipeline已终止", "error")
            logger.info(f"pipeline已终止")
            raise asyncio.CancelledError("Pipeline terminated by user")

        # 判断是否使用RAG
        use_rag = rag_mode == "auto" and rag_enabled
        if use_rag:
            # 使用RAG增强一致性检查
            retrieval_results = []
            res = check_entity_consistency_with_enhancement_local(
                chain=entity_consistency_check_chain,
                entity=ent,
                retrieval_results=retrieval_results,
                model_name=args.model_name,
                model_path=args.model_path,
                device=args.device,
                quantization=args.llm_quantization
            )
        else:
            res = check_entity_consistency_local(
                chain=entity_consistency_check_chain,
                entity=ent,
                model_name=args.model_name,
                model_path=args.model_path,
                device=args.device,
                quantization=args.llm_quantization
            )

        results.append(res)
        await log_callback(f"检查实体 {ent.entity_id} 一致性: {res}")
        logger.info(f"检查实体 {ent.entity_id} 一致性: {res}")

    await log_callback(f"完成检查实体一致性")
    logger.info(f"完成检查实体一致性")

    # 根据检查结果进行修改（简化版）
    await log_callback(f"开始修正实体一致性")
    logger.info(f"开始修正实体一致性")
    res_list = []
    # 对输入的实体进行剔除，只保留冲突实体
    conflict_ents = [ent for ent in results if ent.get("has_conflict") is True]
    logger.info(f"冲突实体: {conflict_ents}")

    consistency_correct_chain = get_consistency_correct_chain_local(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        quantization=args.llm_quantization
    )

    # 对每个chunk进行修正（简化版：直接返回原文本）
    for chunk in chunks:
        chunk_input = f"原始文本:{chunk}\n实体冲突分析结果:{results}"
        res = consistency_correct_chain.invoke(chunk_input).content
        logger.info(f"段落修正结果: \n{res}")
        res_dict = {
            "original_text": chunk,
            "corrected_text": res
        }
        await log_callback(f"段落修正结果: \n{res}")
        logger.info(f"段落修正结果: \n{res}")
        res_list.append(res_dict)

    return res_list

async def extract_entities_with_rag_local(chain, text, log_callback, args):
    """本地版本：使用RAG辅助提取实体"""
    global rag_components_local

    try:
        # 1. 检查是否有RAG组件
        if 'router' not in rag_components_local:
            return extract_entities_local(
                chain=chain,
                text=text,
                model_name=args.model_name,
                model_path=args.model_path,
                device=args.device,
                quantization=args.llm_quantization
            )

        router = rag_components_local['router']
        retriever = rag_components_local['retriever']

        # 2. 路由决策
        decision = router.invoke({"query": text}).content.strip()
        await log_callback(f"RAG路由决策: {decision}")

        if "retrieve" in decision.lower():
            # 执行检索
            docs = retriever.retrieve_local(text)

            # 展平所有文档
            all_docs = []
            for docs_list in docs:
                all_docs.extend(docs_list)

            # 发送检索结果
            await log_callback(f"RAG检索到 {len(all_docs)} 个相关文档")

            # 3. 使用检索到的文档辅助提取
            context = "\n".join(all_docs[:5])  # 取前5个文档

            enhanced_text = f"""
原始文本: {text}

相关上下文:
{context}

请基于以上信息提取实体。
"""

            # 使用增强后的文本提取实体
            return extract_entities_local(
                chain=chain,
                text=enhanced_text,
                model_name=args.model_name,
                model_path=args.model_path,
                device=args.device,
                quantization=args.llm_quantization
            )

    except Exception as e:
        await log_callback(f"RAG辅助提取失败: {e}，使用直接提取")
        return extract_entities_local(
            chain=chain,
            text=text,
            model_name=args.model_name,
            model_path=args.model_path,
            device=args.device,
            quantization=args.llm_quantization
        )

    return extract_entities_local(
        chain=chain,
        text=text,
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        quantization=args.llm_quantization
    )

async def run_grammar_pipeline_local(text: str, args, log_callback, **kwargs):
    """本地版本的语法纠错pipeline"""
    logger = kwargs.get("logger", logging.getLogger(__name__))
    cancellation_token = kwargs.get("cancellation_token", None)
    rag_mode = kwargs.get("rag_mode", "none")
    rag_enabled = kwargs.get("rag_enabled", False)

    # chain获取（本地版本）
    grammar_check_chain = get_grammar_check_chain_local(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        quantization=args.llm_quantization
    )

    # chunking 文本
    await log_callback(f"文本长度: {len(text)}")
    logger.info(f"文本长度: {len(text)}")
    chunks = chunking(text, chunk_size=128)

    # 对每个chunk进行语法检查
    await log_callback(f"开始对 {len(chunks)} 个chunk进行语法检查")
    logger.info(f"开始对 {len(chunks)} 个chunk进行语法检查")
    grammar_results = []
    for chunk in chunks:

        # 检查是否需要终止
        if cancellation_token and cancellation_token.is_set():
            await log_callback(f"pipeline已终止", "error")
            logger.info(f"pipeline已终止")
            raise asyncio.CancelledError("Pipeline terminated by user")

        # 使用RAG增强语法检查
        use_rag = rag_mode == "auto" and rag_enabled
        if use_rag:
            result = await run_grammar_check_with_rag_local(grammar_check_chain, chunk, log_callback, args)
        else:
            result = grammar_check_chain.invoke({"new_message": chunk}).content

        result_dict = json.loads(result)
        result_dict["original_text"] = chunk
        grammar_results.append(result_dict)

        await log_callback(f"语法检查结果: {result_dict}")
        logger.info(f"语法检查结果: {result_dict}")

    await log_callback(f"语法检查完成，共检查 {len(chunks)} 个chunk")
    logger.info(f"语法检查完成，共检查 {len(chunks)} 个chunk")

    await log_callback(f"语法纠错完成")
    logger.info(f"语法纠错完成")

    return grammar_results

async def run_grammar_check_with_rag_local(chain, text, log_callback, args):
    """本地版本：使用RAG增强语法检查"""
    global rag_components_local

    # 检查是否有RAG组件
    if 'router' not in rag_components_local:
        return chain.invoke({"new_message": text}).content

    router = rag_components_local['router']
    retriever = rag_components_local['retriever']

    # 路由决策
    decision = router.invoke({"query": text}).content.strip()
    await log_callback(f"RAG路由决策: {decision}")

    if "retrieve" in decision.lower():
        # 执行检索
        docs = retriever.retrieve_local(text)

        # 展平所有文档
        all_docs = []
        for docs_list in docs:
            all_docs.extend(docs_list)

        # 发送检索结果
        await log_callback(f"RAG检索到 {len(all_docs)} 个相关文档")

        # 使用检索到的文档辅助纠错
        context = "\n".join(all_docs[:5])

        enhanced_text = f"""
原始文本: {text}

相关上下文:
{context}

请基于以上信息进行语法纠错，保持原意不变。
"""

        return chain.invoke({"new_message": enhanced_text}).content

    # 直接纠错
    return chain.invoke({"new_message": text}).content

# 处理反馈的函数（保持不变）
def process_feedback_local(feedback_data, args, logger):
    """处理用户反馈的同步函数（本地版本）"""
    try:
        pipeline = feedback_data.get("pipeline")
        results = feedback_data.get("results")
        rating = feedback_data.get("rating")
        comment = feedback_data.get("comment", "")

        if not pipeline or not results:
            logger.error("无效的反馈数据：缺少pipeline或results")
            return "无效的反馈数据"

        # 构造用户反馈字符串
        user_feedback = f"评分: {rating}/5"
        if comment:
            user_feedback += f"\n评论: {comment}"

        # 根据pipeline类型调用相应的反馈处理函数
        if pipeline == "consistency":
            # 替换原有函数中的input()调用，直接使用用户提交的反馈
            import builtins
            original_input = builtins.input
            builtins.input = lambda _=None: user_feedback

            try:
                summary = collect_consistency_feedback(results, args.log_dir, args, logger)
            finally:
                builtins.input = original_input

            return f"一致性检测反馈已提交。反馈总结: {summary}"

        elif pipeline == "grammar":
            # 替换原有函数中的input()调用，直接使用用户提交的反馈
            import builtins
            original_input = builtins.input
            builtins.input = lambda _=None: user_feedback

            try:
                summary = collect_grammar_feedback(results, args.log_dir, args, logger)
            finally:
                builtins.input = original_input

            return f"语法纠错反馈已提交。反馈总结: {summary}"

        else:
            logger.error(f"未知的pipeline类型：{pipeline}")
            return "未知的pipeline类型"

    except Exception as e:
        logger.exception("处理反馈时发生错误")
        return f"处理反馈时发生错误：{str(e)}"

router = APIRouter()

# RAG状态管理（本地版本）
rag_indices_local = {}
rag_components_local = {}

@router.websocket("/ws/chat_local")
async def websocket_chat_local(websocket: WebSocket):
    await websocket.accept()
    cancellation_token = None
    try:
        while True:
            data = await websocket.receive_json()
            logger = websocket.app.state.logger

            # 处理反馈请求
            if data.get("action") == "feedback":
                logger.info("收到用户反馈请求")
                args = websocket.app.state.args

                # 调用同步反馈处理函数
                feedback_result = process_feedback_local(data, args, logger)

                # 发送反馈结果
                await websocket.send_json({"feedback_result": feedback_result})
                continue

            # 处理正常的检测请求
            message = data.get("message")
            file_info = data.get("file")  # dict {filename, content}
            pipeline = data.get("action", "consistency")  # 从action字段获取操作类型，默认使用一致性检测pipeline
            rag_mode = data.get("rag_mode", "none")  # none, mandatory, auto
            rag_enabled = rag_mode != "none"

            # 将前端发来的 base64 文件转成 UploadFile
            file = None
            if file_info:
                filename = file_info["filename"]
                content_base64 = file_info["content"].split(",")[-1]  # 去掉 data:*/*;base64,
                file_bytes = base64.b64decode(content_base64)
                file = UploadFile(filename=filename, file=io.BytesIO(file_bytes))

            args = websocket.app.state.args
            logger = websocket.app.state.logger

            # 创建取消令牌
            cancellation_token = asyncio.Event()

            text = get_text_from_input(message, file)
            if not text.strip():
                await websocket.send_json({"error": "未提供消息或文件"})
                continue

            async def log_callback(msg, msg_type="log"):
                await websocket.send_json({"log": msg, "type": msg_type})

            # 加载或初始化RAG组件（本地版本）
            if rag_enabled:
                await init_rag_components_local(args, rag_mode, log_callback)
            else:
                logger.info("本地RAG功能未启用")

            # 根据选择的pipeline执行相应的函数
            if pipeline == "consistency":
                results = await run_consistency_pipeline_local(
                    text,
                    args,
                    log_callback,
                    rag_mode=rag_mode,
                    rag_enabled=rag_enabled,
                    logger=logger,
                    cancellation_token=cancellation_token
                )
            else:
                results = await run_grammar_pipeline_local(
                    text,
                    args,
                    log_callback,
                    rag_mode=rag_mode,
                    rag_enabled=rag_enabled,
                    logger=logger,
                    cancellation_token=cancellation_token
                )

            await websocket.send_json({"results": results, "done": True, "pipeline": pipeline})

    except WebSocketDisconnect:
        # WebSocket连接断开时，设置取消令牌
        if cancellation_token:
            cancellation_token.set()
        logger.info("WebSocket连接已断开")
    except asyncio.CancelledError:
        logger.info("Pipeline执行被用户终止")
    except Exception as e:
        logger.exception(e)
        await websocket.send_json({"error": str(e)})
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
