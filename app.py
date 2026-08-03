import json
import uuid
import io
import httpx
import fitz
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from langchain.messages import HumanMessage, AIMessage
from agent_setup import create_research_agent, set_current_thread
from database import init_db, create_session, list_sessions, delete_session, update_session_title
from schemas import ChatRequest, HistoryItem, HistoryResponse, DeleteResponse
from oss_sts import get_oss_token
from paper_parser import extract_text_from_bytes, detect_sections, chunk_sections, guess_title
from rag_store import get_rag

@asynccontextmanager
async def lifespan(app):
    init_db()
    app.state.research_agent = create_research_agent()
    yield


app = FastAPI(
    title="AI Research Copilot",
    description="AI 科研论文助手：上传 PDF，自动生成结构化摘要，支持论文问答",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


async def download_from_oss(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


@app.post("/chat")
async def chat(request: ChatRequest, fastapi_req: Request):
    agent = fastapi_req.app.state.research_agent
    rag = get_rag()

    if not request.thread_id:
        thread_id = str(uuid.uuid4())
        create_session(thread_id, "新论文")
        is_new = True
    else:
        thread_id = request.thread_id
        is_new = False

    # 设置当前线程 ID（供 Agent 工具使用）
    set_current_thread(thread_id)

    # 处理 PDF 上传
    paper_title = ""
    if request.file_type == "pdf" and request.image_url:
        pdf_data = await download_from_oss(request.image_url)
        full_text, line_pages = extract_text_from_bytes(pdf_data)
        sections, section_pages = detect_sections(full_text, line_pages)
        chunks = chunk_sections(sections, section_pages)
        paper_title = guess_title(full_text)
        rag.store(thread_id, chunks, paper_title)

        # 更新会话标题为论文名
        if paper_title:
            update_session_title(thread_id, paper_title)

    # 构建消息
    user_text = request.user_text or "请帮我分析这篇论文"

    if request.file_type == "pdf" and paper_title:
        text_content = f"[用户上传了新论文]《{paper_title}》\n用户说：{user_text}\n\n请先阅读论文内容，然后生成结构化摘要。"
    else:
        text_content = user_text

    message = HumanMessage(content=[{"type": "text", "text": text_content}])

    def event_stream():
        yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id})}\n\n"

        ai_text = ""
        config = {"configurable": {"thread_id": thread_id}}
        for chunk, metadata in agent.stream(
            {"messages": [message]},
            stream_mode="messages",
            config=config,
        ):
            if hasattr(chunk, "content") and chunk.content:
                # 过滤工具消息，只保留 AI 文字输出
                from langchain.messages import AIMessageChunk
                if isinstance(chunk, AIMessageChunk):
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if text:
                        ai_text += text
                        yield f"data: {json.dumps({'type': 'content', 'content': text})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

        # 新会话自动标题
        if is_new and ai_text and not paper_title:
            title = ai_text.split("\n")[0].strip()[:30]
            title = title.replace("#", "").replace("*", "").strip()
            if title:
                update_session_title(thread_id, title)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/history", response_model=HistoryResponse)
async def get_history():
    sessions = list_sessions()
    items = [HistoryItem(**s) for s in sessions]
    return HistoryResponse(sessions=items)


@app.delete("/history/{thread_id}", response_model=DeleteResponse)
async def clear_history(thread_id: str):
    rag = get_rag()
    rag.delete_collection(thread_id)
    success = delete_session(thread_id)
    return DeleteResponse(message="清空成功" if success else "清空失败")


@app.get("/oss/token")
async def oss_token():
    return get_oss_token()


@app.get("/chat/{thread_id}/messages")
async def refresh(thread_id: str, fastapi_req: Request) -> list[dict]:
    agent = fastapi_req.app.state.research_agent
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    messages = []
    for msg in state.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append(part["text"])
                content = " ".join(text_parts)
            messages.append({"role": "user", "content": content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    return messages


app.mount("/", StaticFiles(directory="static", html=True), name="static")
