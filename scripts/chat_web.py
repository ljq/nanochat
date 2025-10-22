#!/usr/bin/env python3
"""
统一的web聊天服务器 - 从单个FastAPI实例提供UI和API。

使用数据并行性将请求分发到多个GPU。每个GPU加载
模型的完整副本，传入的请求被分发到可用的工作器。

启动示例：

- 单个可用GPU（默认）
python -m scripts.chat_web

- 4个GPU
python -m scripts.chat_web --num-gpus 4

要聊天，请打开控制台中打印的URL。（如果在云盒上，请确保使用公共IP）

端点：
  GET  /           - 聊天UI
  POST /chat/completions - 聊天API（仅流式）
  GET  /health     - 健康检查和工作器池状态
  GET  /stats      - 工作器池统计信息和GPU利用率

滥用预防：
  - 每个请求最多500条消息
  - 每条消息最多8000个字符
  - 总对话长度最多32000个字符
  - 温度限制在0.0-2.0
  - Top-k限制在1-200
  - 最大token数限制在1-4096
"""

import argparse
import json
import os
import torch
import asyncio
import logging
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass

from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# 滥用预防限制
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 1
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = argparse.ArgumentParser(description='NanoChat Web Server')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|mid|rl")
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Default temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Default max tokens for generation')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
args = parser.parse_args()

# 配置对话流量的日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()

@dataclass
class Worker:
    """在特定GPU上加载模型的工作器。"""
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object
    autocast_ctx: torch.amp.autocast

class WorkerPool:
    """工作器池，每个工作器在不同的GPU上有模型副本。"""

    def __init__(self, num_gpus: Optional[int] = None):
        self.num_gpus = num_gpus if num_gpus is not None else torch.cuda.device_count()
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """在每个GPU上加载模型。"""
        print(f"正在初始化具有 {self.num_gpus} 个GPU的工作器池...")

        for gpu_id in range(self.num_gpus):
            device = torch.device(f"cuda:{gpu_id}")
            print(f"在GPU {gpu_id}上加载模型...")

            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
                autocast_ctx=autocast_ctx
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"所有 {self.num_gpus} 个工作器已初始化！")

    async def acquire_worker(self) -> Worker:
        """从池中获取一个可用的工作器。"""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """将工作器返回到池中。"""
        await self.available_workers.put(worker)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None

def validate_chat_request(request: ChatRequest):
    """验证聊天请求以防止滥用。"""
    # 检查消息数量
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="至少需要一条消息")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"消息过多。每个请求最多允许 {MAX_MESSAGES_PER_REQUEST} 条消息"
        )

    # 检查单个消息长度和总对话长度
    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"消息 {i} 内容为空")

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"消息 {i} 太长。每条消息最多允许 {MAX_MESSAGE_LENGTH} 个字符"
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"总对话太长。最多允许 {MAX_TOTAL_CONVERSATION_LENGTH} 个字符"
        )

    # 验证角色值
    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"消息 {i} 角色无效。必须是 'user'、'assistant' 或 'system'"
            )

    # 验证温度
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"温度必须在 {MIN_TEMPERATURE} 和 {MAX_TEMPERATURE} 之间"
            )

    # 验证top_k
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k 必须在 {MIN_TOP_K} 和 {MAX_TOP_K} 之间"
            )

    # 验证max_tokens
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens 必须在 {MIN_MAX_TOKENS} 和 {MAX_MAX_TOKENS} 之间"
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """在启动时在所有GPU上加载模型。"""
    print("正在跨GPU加载nanochat模型...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    print(f"服务器准备就绪：http://localhost:{args.port}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """提供聊天UI。"""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r") as f:
        html_content = f.read()
    # 替换API_URL以使用相同源
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    """提供NanoChat徽标用于favicon和标题。"""
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")

async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """生成助手响应（流式）。"""
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    # 累积token以正确处理多字节UTF-8字符（如表情符号）
    accumulated_tokens = []
    # 跟踪最后一个完整的UTF-8字符串（没有替换字符）
    last_clean_text = ""

    with worker.autocast_ctx:
        for token_column, token_masks in worker.engine.generate(
            tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=random.randint(0, 2**31 - 1)
        ):
            token = token_column[0]

            # 停止条件
            if token == assistant_end or token == bos:
                break

            # 将token附加到序列
            accumulated_tokens.append(token)
            # 解码所有累积的token以获得正确的UTF-8处理
            # 注意解码是一个相当高效的操作，基本上是表查找和字符串连接
            current_text = worker.tokenizer.decode(accumulated_tokens)
            # 仅在不以替换字符结尾时发出文本
            # 这确保我们不会发出不完整的UTF-8序列
            if not current_text.endswith(''):
                # 仅提取自上次干净解码以来的新文本
                new_text = current_text[len(last_clean_text):]
                if new_text:  # 仅在有新内容时生成
                    yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                    last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """聊天完成端点（仅流式）- 使用工作器池进行多GPU处理。"""

    # 基本验证以防止滥用
    validate_chat_request(request)

    # 将传入对话记录到控制台
    logger.info("="*20)
    for i, message in enumerate(request.messages):
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-"*20)

    # 从池中获取工作器（如果所有工作器都忙，将等待）
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        # 构建对话token
        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<|user_start|>")
        user_end = worker.tokenizer.encode_special("<|user_end|>")
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

        conversation_tokens = [bos]
        for message in request.messages:
            if message.role == "user":
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)

        conversation_tokens.append(assistant_start)

        # 流式响应，完成后释放工作器
        response_tokens = []
        async def stream_and_release():
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k
                ):
                    # 累积响应用于日志记录
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    yield chunk
            finally:
                # 将助手响应记录到控制台
                full_response = "".join(response_tokens)
                logger.info(f"[助手] (GPU {worker.gpu_id}): {full_response}")
                logger.info("="*20)
                # 流式传输完成后将工作器释放回池中
                await worker_pool.release_worker(worker)

        return StreamingResponse(
            stream_and_release(),
            media_type="text/event-stream"
        )
    except Exception as e:
        # 确保即使在错误时也释放工作器
        await worker_pool.release_worker(worker)
        raise e

@app.get("/health")
async def health():
    """健康检查端点。"""
    worker_pool = getattr(app.state, 'worker_pool', None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0
    }

@app.get("/stats")
async def stats():
    """获取工作器池统计信息。"""
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {
                "gpu_id": w.gpu_id,
                "device": str(w.device)
            } for w in worker_pool.workers
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print(f"启动NanoChat Web服务器")
    print(f"温度: {args.temperature}, Top-k: {args.top_k}, 最大token数: {args.max_tokens}")
    uvicorn.run(app, host=args.host, port=args.port)
