from llm.model import get_grammar_check_chain_with_memory, get_grammar_check_chain, get_entity_extract_chain, get_entity_consistency_check_chain, get_memory_summary_chain
from llm.entity import EntityStore, extract_entities, summarize_entity_memory, check_entity_consistency
from filereader.reader import chunking, get_text_from_input
import io
from fastapi import APIRouter, UploadFile, File, Form, Request, WebSocket, WebSocketDisconnect
import json
import base64
import asyncio
import uuid

# 全局缓存日志和结果
TASKS = {}  # task_id -> {"logs": [], "result": None, "done": False}

async def run_pipeline(text: str, args, log_callback, **kwargs):
    # 初始化模型
    grammar_check_with_memory = get_grammar_check_chain_with_memory(args.model_name, args.base_url)
    entity_extract_chain = get_entity_extract_chain(args.model_name, args.base_url)
    entity_consistency_check_chain = get_entity_consistency_check_chain(args.model_name, args.base_url)
    memory_summary_chain = get_memory_summary_chain(args.model_name, args.base_url)
    
    logger = kwargs.get("logger")

    await log_callback(f"开始运行pipeline，模型: {args.model_name}")
    # chunking 文本
    await log_callback(f"文本长度: {len(text)}")
    chunks = chunking(text)
    ent_store = EntityStore()
    # 处理每个 chunk
    previous_memory = ""
    for i, chunk in enumerate(chunks):
        chunk_input = (
            f"前文要点总结:{previous_memory}\n当前输入文本:{chunk}"
            if previous_memory else chunk
        )

        ents = extract_entities(entity_extract_chain, chunk_input)
        for ent in ents:
            ent_store.add_entity(ent)
        await log_callback(f"第 {i+1} 个 chunk 提取实体: {ents}")
        
        if i < len(chunks) - 1:
            previous_memory = summarize_entity_memory(
                memory_summary_chain, chunk_input
            )
    # 检查实体一致性
    await log_callback(f"实体总数: {len(ent_store.all_entities())}")
    await log_callback(f"开始检查实体一致性")     
    results = []
    for ent in ent_store.all_entities():
        res = check_entity_consistency(
            entity_consistency_check_chain, ent
        )
        results.append(res)
        await log_callback(f"检查实体 {ent.entity_id} 一致性: {res}")

    await log_callback(f"完成检查实体一致性")     
    return results


router = APIRouter()

@router.post("/chat")
async def chat(
        request: Request,
        message: str = Form(None),
        history: str = Form("[]"),
        file: UploadFile = File(None)
):
    """
    Chat 层：只做三件事
    1. 统一输入
    2. 决定要不要跑 pipeline
    3. 把结果变成人话
    """
    history = json.loads(history)
    logger = request.app.state.logger
    args = request.app.state.args
    # 1. 统一输入
    text = get_text_from_input(message, file)
    if not text.strip():
        return {
            "reply": "请提供文本或上传 docx/pdf 文件。",
            "history": history
        }
    # 2. 决定要不要跑pipeline
    results = run_pipeline(text, args, logger=logger)

    # 3. 把结果变成人话
    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": "\n".join(str(r) for r in results["results"])})
    return {
        "reply": "\n".join(str(r) for r in results["results"]),
        "history": history,
        "logs": results["logs"]
    }

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        message = data.get("message")
        file_info = data.get("file")  # dict {filename, content}

        # 将前端发来的 base64 文件转成 UploadFile
        file = None
        if file_info:
            filename = file_info["filename"]
            content_base64 = file_info["content"].split(",")[-1]  # 去掉 data:*/*;base64,
            file_bytes = base64.b64decode(content_base64)
            file = UploadFile(filename=filename, file=io.BytesIO(file_bytes))

        args = websocket.app.state.args
        logger = websocket.app.state.logger

        text = get_text_from_input(message, file)
        if not text.strip():
            await websocket.send_json({"error": "未提供消息或文件"})
            return

        async def log_callback(msg):
            await websocket.send_json({"log": msg})

        results = await run_pipeline(text, args, log_callback)
        await websocket.send_json({"results": results, "done": True})

    except Exception as e:
        logger.exception(e)
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()


