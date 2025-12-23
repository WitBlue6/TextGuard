from llm.model import get_grammar_check_chain_with_memory, get_grammar_check_chain, get_entity_extract_chain, get_entity_consistency_check_chain, get_memory_summary_chain, get_consistency_correct_chain, get_feedback_summary_chain
from llm.entity import EntityStore, extract_entities, summarize_entity_memory, check_entity_consistency
from filereader.reader import chunking, get_text_from_input
from feedback import collect_consistency_feedback, collect_grammar_feedback

import io
from fastapi import APIRouter, UploadFile, File, Form, Request, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import json
import base64
import asyncio
import uuid
import logging

# 全局缓存日志和结果
TASKS = {}  # task_id -> {"logs": [], "result": None, "done": False}

async def run_consistency_pipeline(text: str, args, log_callback, **kwargs):
    # 初始化模型
    grammar_check_with_memory = get_grammar_check_chain_with_memory(args.model_name, args.base_url)
    entity_extract_chain = get_entity_extract_chain(args.model_name, args.base_url)
    entity_consistency_check_chain = get_entity_consistency_check_chain(args.model_name, args.base_url)
    memory_summary_chain = get_memory_summary_chain(args.model_name, args.base_url)
    
    logger = kwargs.get("logger", logging.getLogger(__name__))
    cancellation_token = kwargs.get("cancellation_token", None)

    await log_callback(f"开始运行一致性检测pipeline，模型: {args.model_name}")
    logger.info(f"开始运行一致性检测pipeline，模型: {args.model_name}")
    # chunking 文本
    await log_callback(f"文本长度: {len(text)}")
    logger.info(f"文本长度: {len(text)}")
    chunks = chunking(text)
    ent_store = EntityStore()
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

        ents = extract_entities(entity_extract_chain, chunk_input)
        for ent in ents:
            ent_store.add_entity(ent)
        await log_callback(f"第 {i+1} 个 chunk 提取实体: {ents}")
        logger.info(f"第 {i+1} 个 chunk 提取实体: {ents}")
        
        if i < len(chunks) - 1:
            previous_memory = summarize_entity_memory(
                memory_summary_chain, chunk_input
            )
    # 检查实体一致性
    await log_callback(f"实体总数: {len(ent_store.all_entities())}")
    logger.info(f"实体总数: {len(ent_store.all_entities())}")
    await log_callback(f"开始检查实体一致性")     
    logger.info(f"开始检查实体一致性")     
    results = []
    for ent in ent_store.all_entities():
        # 定期检查是否有取消请求
        await asyncio.sleep(0.1)

        # 检查是否需要终止
        if cancellation_token and cancellation_token.is_set():
            await log_callback(f"pipeline已终止", "error")
            logger.info(f"pipeline已终止")
            raise asyncio.CancelledError("Pipeline terminated by user")
            
        res = check_entity_consistency(
            entity_consistency_check_chain, ent
        )
        results.append(res)
        await log_callback(f"检查实体 {ent.entity_id} 一致性: {res}")
        logger.info(f"检查实体 {ent.entity_id} 一致性: {res}")

    await log_callback(f"完成检查实体一致性")     
    logger.info(f"完成检查实体一致性")   

    # 根据检查结果进行修改
    await log_callback(f"开始修正实体一致性")     
    logger.info(f"开始修正实体一致性")     
    res_list = []
    # 对输入的实体进行剔除，只保留冲突实体
    conflict_ents = [ent for ent in results if ent["has_conflict"] is True]
    logger.info(f"冲突实体: {conflict_ents}")
    consistency_correct_chain = get_consistency_correct_chain(args.model_name, args.base_url)  
    # 对每个chunk进行修正
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

async def run_grammar_pipeline(text: str, args, log_callback, **kwargs):
    """新的语法纠错pipeline"""
    logger = kwargs.get("logger", logging.getLogger(__name__))
    cancellation_token = kwargs.get("cancellation_token", None)
    # chain获取
    grammar_check_chain = get_grammar_check_chain(args.model_name, args.base_url)
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
    
    # 返回模拟结果
    return grammar_results

# 处理反馈的函数
def process_feedback(feedback_data, args, logger):
    """处理用户反馈的同步函数"""
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

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
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
                feedback_result = process_feedback(data, args, logger)
                
                # 发送反馈结果
                await websocket.send_json({"feedback_result": feedback_result})
                continue
            
            # 处理正常的检测请求
            message = data.get("message")
            file_info = data.get("file")  # dict {filename, content}
            pipeline = data.get("pipeline", "consistency")  # 默认使用一致性检测pipeline

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

            # 根据选择的pipeline执行相应的函数
            if pipeline == "consistency":
                results = await run_consistency_pipeline(
                    text, 
                    args, 
                    log_callback, 
                    logger=logger,
                    cancellation_token=cancellation_token
                )
            else:
                results = await run_grammar_pipeline(
                    text, 
                    args, 
                    log_callback, 
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