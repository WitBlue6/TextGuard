from llm.model import get_grammar_check_chain_with_memory, get_grammar_check_chain, get_entity_extract_chain, get_entity_consistency_check_chain, get_memory_summary_chain
from llm.entity import EntityStore, extract_entities, summarize_entity_memory, check_entity_consistency
from filereader.reader import chunking, get_text_from_input
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
    return results

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


router = APIRouter()

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    cancellation_token = None
    try:
        data = await websocket.receive_json()
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
            return

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