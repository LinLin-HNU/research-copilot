import json
import uuid
import io
import httpx
import fitz
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from langchain.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langchain_core.messages import RemoveMessage, ToolMessageChunk
from agent_setup import create_research_agent, set_current_thread, reset_retrieval_memory
from database import init_db, create_session, list_sessions, delete_session, update_session_title
from schemas import ChatRequest, HistoryItem, HistoryResponse, DeleteResponse
from oss_sts import get_oss_token
from paper_parser import extract_text_from_bytes, detect_sections, chunk_sections, guess_title
from rag_store import get_rag
from query_router import build_route

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


SUMMARY_QUERIES = [
    "abstract research problem and core contribution",
    "method architecture algorithm model design",
    "experiments datasets metrics results ablation",
    "conclusion limitations future work",
]


def build_evidence(rag, thread_id: str, queries: list[str], per_query: int = 2) -> str:
    """按查询列表在 Python 侧预检索论文证据，内容去重后格式化为
    【证据 #N | 来源: Section | 第 X-Y 页】，注入消息让 Agent 一步生成，
    避免反复调用检索工具导致递归超限 / token 空耗。"""
    seen = set()
    parts = []
    n = 1
    for q in queries:
        for r in rag.search(thread_id, q, k=per_query):
            if r["content"] in seen:
                continue
            seen.add(r["content"])
            section = r["section"].title()
            content = r["content"][:600]
            page = f" | 第 {r.get('page_start', 0)}-{r.get('page_end', 0)} 页" if r.get("page_end", 0) else ""
            parts.append(f"【证据 #{n} | 来源: {section}{page}】\n{content}")
            n += 1
    return "\n\n---\n\n".join(parts)


ROUTE_INSTRUCTIONS = {
    "A1": (
        "请基于以上证据回答。结论中的一级事实（贡献/方法/实验数据/结论）必须附【来源章节】+【原文引用】；"
        "证据不足的部分明确标注'论文中未检索到'。不要调用检索工具。"
    ),
    "A2": (
        "请区分两类：论文明确表述的不足（附【来源】+【原文引用】）；"
        "从论文设计推导出的潜在问题（标注'属从论文设计推导，论文未明确提及'）。不要调用检索工具。"
    ),
    "C": (
        "请结合以上证据与你的领域知识回答。引用论文处标【来源】；"
        "领域知识标注'（非论文内容，属领域背景）'；你的推断标注'（基于领域知识的推测）'。不要调用检索工具。"
    ),
}


def route_user_question(rag, thread_id: str, text: str) -> str:
    """非上传问题的路由：决定是否需要检索，构造注入证据/指令的用户消息。"""
    route = build_route(text)
    category = route["category"]

    if not route["needs_retrieval"]:
        return (
            f"用户问题：{text}\n\n"
            "路由判定：该问题属于领域知识类（B），不需要检索上传的论文。"
            "请直接基于你的已有知识回答，并在回答中显式标注："
            "'以下内容来自模型自身知识，不来自上传论文'。不要调用检索工具。"
        )

    evidence = build_evidence(rag, thread_id, route["queries"])
    if not evidence:
        return (
            f"用户问题：{text}\n\n"
            "路由判定：该问题需要论文内容，但当前会话没有可用的论文证据（可能尚未上传论文）。"
            "请礼貌提示用户先上传论文 PDF，不要用外部知识代替。不要调用检索工具。"
        )

    instruction = ROUTE_INSTRUCTIONS.get(category, ROUTE_INSTRUCTIONS["A1"])
    return (
        f"用户问题：{text}\n\n"
        f"以下是从论文中检索到的相关证据（与 retrieve_paper 相同格式）：\n\n"
        f"{evidence}\n\n"
        f"{instruction}"
    )


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

    # 设置当前线程 ID（供 Agent 工具使用），并重置该会话的检索记忆
    set_current_thread(thread_id)
    reset_retrieval_memory(thread_id)

    # 处理 PDF 上传
    paper_title = ""
    if request.file_type == "pdf" and request.image_url:
        # V1 语义：一个会话只装一篇论文，新上传 = 替换旧论文（避免多篇 chunk 混在一个集合）
        rag.delete_collection(thread_id)
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
        evidence = build_evidence(rag, thread_id, SUMMARY_QUERIES)
        if evidence:
            text_content = (
                f"[用户上传了新论文]《{paper_title}》\n"
                f"用户说：{user_text}\n\n"
                f"以下是从论文中检索到的关键章节证据（与 retrieve_paper 相同格式），"
                f"请直接基于这些证据生成结构化摘要，不要调用检索工具：\n\n"
                f"{evidence}"
            )
        else:
            text_content = f"[用户上传了新论文]《{paper_title}》\n用户说：{user_text}\n\n请先阅读论文内容，然后生成结构化摘要。"
    else:
        text_content = route_user_question(rag, thread_id, user_text)

    message = HumanMessage(content=[{"type": "text", "text": text_content}])
    config = {"recursion_limit": 15, "configurable": {"thread_id": thread_id}}

    # 堵住 token 炸弹：清理会话历史里的工具调用记录，避免大段检索结果在后续请求中反复重发
    try:
        state = agent.get_state(config)
        to_remove = [
            RemoveMessage(id=m.id)
            for m in state.values.get("messages", [])
            if isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.tool_calls)
        ]
        if to_remove:
            agent.update_state(config, {"messages": to_remove})
    except Exception:
        pass

    def event_stream():
        yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id})}\n\n"

        ai_text = ""
        tool_active = False
        for chunk, metadata in agent.stream(
            {"messages": [message]},
            stream_mode="messages",
            config=config,
        ):
            if isinstance(chunk, AIMessageChunk):
                # 检测到工具调用 → 发进度事件，避免"烧 token 无反应"
                if getattr(chunk, "tool_call_chunks", None) and not tool_active:
                    tool_active = True
                    yield f"data: {json.dumps({'type': 'status', 'message': '正在检索论文内容...'})}\n\n"
                text = chunk.content if isinstance(chunk.content, str) else ""
                if text:
                    ai_text += text
                    yield f"data: {json.dumps({'type': 'content', 'content': text})}\n\n"
            elif isinstance(chunk, (ToolMessage, ToolMessageChunk)):
                if tool_active:
                    tool_active = False
                    yield f"data: {json.dumps({'type': 'status', 'message': '检索完成，正在生成回答...'})}\n\n"

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
