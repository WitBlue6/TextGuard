from llm.model import get_grammar_check_chain_with_memory, get_grammar_check_chain, get_entity_extract_chain, get_entity_consistency_check_chain, get_memory_summary_chain
from llm.entity import EntityStore, extract_entities, summarize_entity_memory, check_entity_consistency
from filereader.reader import chunking, get_text_from_input
from io import BytesIO
from fastapi import APIRouter, UploadFile, File, Form, Request, WebSocket, WebSocketDisconnect
import json
import base64
import asyncio
import uuid

# 全局缓存日志和结果
TASKS = {}  # task_id -> {"logs": [], "result": None, "done": False}

async def run_pipeline(text: str, task_id, args, **kwargs):
    # 初始化模型
    grammar_check_with_memory = get_grammar_check_chain_with_memory(args.model_name, args.base_url)
    entity_extract_chain = get_entity_extract_chain(args.model_name, args.base_url)
    entity_consistency_check_chain = get_entity_consistency_check_chain(args.model_name, args.base_url)
    memory_summary_chain = get_memory_summary_chain(args.model_name, args.base_url)
    
    logger = kwargs.get("logger")
    log_messages = []
    def log(msg):
        logger.info(msg)
        TASKS[task_id]["logs"].append(msg)

    log(f"开始运行pipeline，模型: {args.model_name}")

    # chunking 文本
    log(f"文本长度: {len(text)}")
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
        log(f"第 {i+1} 个 chunk 提取实体: {ents}")
        
        if i < len(chunks) - 1:
            previous_memory = summarize_entity_memory(
                memory_summary_chain, chunk_input
            )
    # 检查实体一致性
    log(f"实体总数: {len(ent_store.all_entities())}")
    log(f"开始检查实体一致性")     
    results = []
    for ent in ent_store.all_entities():
        res = check_entity_consistency(
            entity_consistency_check_chain, ent
        )
        results.append(res)
        log(f"检查实体 {ent.entity_id} 一致性: {res}")

    log(f"完成检查实体一致性")     
    TASKS[task_id]["result"] = results
    TASKS[task_id]["done"] = True


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

@router.post("/start_task")
async def start_task(request: Request, message: str = Form(None), file: UploadFile = None):
    logger = request.app.state.logger
    args = request.app.state.args

    text = get_text_from_input(message, file)
    if not text.strip():
        return {"error": "请提供文本或上传文件。"}

    task_id = str(uuid.uuid4())
    TASKS[task_id] = {"logs": [], "result": None, "done": False}

    # 后台运行 pipeline
    asyncio.create_task(run_pipeline(text, task_id, args, logger=logger))
    return {"task_id": task_id}

@router.get("/task_status/{task_id}")
async def task_status(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        return {"error": "任务不存在"}
    return {"logs": task["logs"], "done": task["done"], "result": task["result"]}


