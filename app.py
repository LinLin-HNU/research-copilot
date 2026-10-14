import json
import logging
import os
import re
import threading
import uuid
from pathlib import Path
from time import perf_counter
from urllib.parse import unquote

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from langchain.messages import HumanMessage, AIMessage, AIMessageChunk, ToolMessage
from langchain_core.messages import RemoveMessage
from agent_setup import clear_thread_history, get_research_agent, RequestContext
from database import (
    init_db, create_session, list_sessions, delete_session, update_session_title,
    list_adaptive_turns, record_adaptive_turn, record_request_metric, session_exists,
)
from schemas import (
    AdaptiveResearchRequest,
    ChatRequest,
    DeepResearchRequest,
    DeleteResponse,
    HistoryItem,
    HistoryResponse,
)
from deep_research import ChromaPaperRetriever, run_deep_research
from adaptive_research import AdaptivePolicy, public_research_result, run_adaptive_research
from deep_research_runtime import (
    AliyunResearchModel,
    ResearchAccessError,
    load_research_guard,
)
from oss_sts import get_oss_token
from paper_parser import extract_text_from_bytes, detect_sections, chunk_sections, guess_title
from rag_store import get_rag
from query_router import build_route
from trace_admin import TraceRepository


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


app = FastAPI(
    title="AI Research Copilot",
    description="AI 科研论文助手：上传 PDF，自动生成结构化摘要，支持论文问答",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


_research_guard = None
_research_guard_lock = threading.Lock()
MAX_LOCAL_PDF_BYTES = 25 * 1024 * 1024
ADAPTIVE_TRACE_DIR = (
    Path(__file__).parent
    / "benchmark"
    / "experiments"
    / "private"
    / "workflow_adaptive_live"
)


def get_research_guard():
    """Load local experiment limits once; changing .env requires an app restart."""
    global _research_guard
    if _research_guard is None:
        with _research_guard_lock:
            if _research_guard is None:
                _research_guard = load_research_guard()
    return _research_guard


async def download_from_oss(url: str) -> bytes:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


@app.post("/research/import")
async def import_research_pdf(
    request: Request,
    x_research_token: str | None = Header(default=None),
    x_confirm_embedding: str | None = Header(default=None),
    x_paper_filename: str | None = Header(default=None),
):
    """Import a PDF directly from the local client, without generating an answer."""
    try:
        get_research_guard().authenticate(x_research_token)
    except ResearchAccessError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    if x_confirm_embedding != "true":
        raise HTTPException(status_code=400, detail="请先确认论文分块将发送到嵌入服务")
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/pdf":
        raise HTTPException(status_code=415, detail="仅支持 PDF 文件")
    declared_size = request.headers.get("content-length")
    if declared_size and (not declared_size.isdigit() or int(declared_size) > MAX_LOCAL_PDF_BYTES):
        raise HTTPException(status_code=413, detail="PDF 不得超过 25 MB")
    data = bytearray()
    async for part in request.stream():
        data.extend(part)
        if len(data) > MAX_LOCAL_PDF_BYTES:
            raise HTTPException(status_code=413, detail="PDF 不得超过 25 MB")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="文件内容不是有效的 PDF")

    def parse_and_index():
        try:
            full_text, line_pages = extract_text_from_bytes(bytes(data))
            sections, section_pages = detect_sections(full_text, line_pages)
            chunks = chunk_sections(sections, section_pages)
        except Exception as exc:
            raise ValueError("pdf_parse_failed") from exc
        if not chunks:
            raise ValueError("no_readable_paper_text")
        supplied_name = Path(unquote(x_paper_filename or "").replace("\\", "/")).name
        if supplied_name.lower().endswith(".pdf"):
            stem = re.sub(r"^\d{4}[_ -]+", "", supplied_name[:-4])
            title = re.sub(r"\s+", " ", stem.replace("_", " ")).strip()[:100]
        else:
            title = ""
        title = title or guess_title(full_text)
        thread_id = str(uuid.uuid4())
        rag = get_rag()
        try:
            # This calls the configured embedding provider; no generation call is made.
            rag.store(thread_id, chunks, title)
            create_session(thread_id, title)
        except Exception:
            try:
                rag.delete_collection(thread_id)
            except Exception:
                logger.exception("Could not clean up failed import %s", thread_id)
            raise
        return {
            "thread_id": thread_id,
            "title": title,
            "page_count": max(line_pages, default=0),
            "chunk_count": len(chunks),
        }

    try:
        return await run_in_threadpool(parse_and_index)
    except ValueError as exc:
        if str(exc) in {"pdf_parse_failed", "no_readable_paper_text"}:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise
    except Exception as exc:
        logger.exception("Local research PDF import failed")
        raise HTTPException(status_code=502, detail="论文索引失败，请检查嵌入服务和本地日志") from exc


SUMMARY_QUERIES = [
    "abstract research problem and core contribution",
    "method architecture algorithm model design",
    "experiments datasets metrics results ablation",
    "conclusion limitations future work",
]

# 证据预算：单次注入系统提示的证据上限（证据不持久化，但仍需控制当轮 token）
MAX_EVIDENCE_CHARS = 6000
MAX_EVIDENCE_CHUNKS = 8
MAX_CHUNK_CHARS = 600
SEMANTIC_CANDIDATES_PER_QUERY = 8
SEMANTIC_EVIDENCE_CHUNKS = 6
# 不作为论文事实证据的章节（参考文献/附录）
EVIDENCE_SKIP_SECTIONS = {"references", "reference", "appendix", "bibliography"}
# 短于此长度的 chunk 多为孤立标题（如「Background\n2.1」），不作为证据
MIN_EVIDENCE_CHUNK_CHARS = 60


def build_evidence(
    rag,
    thread_id: str,
    queries: list[str],
    per_query: int = 2,
    max_chunks: int = MAX_EVIDENCE_CHUNKS,
    include_candidates: bool = False,
) -> str | tuple[str, list[dict]]:
    """在 Python 侧预检索论文证据，去重 + 预算控制后格式化为
    【证据 #N | 来源: Section | 第 X-Y 页】。

    返回值只通过 RequestContext 进入当轮系统提示，不写进消息历史。
    """
    seen = set()
    parts = []
    n = 1
    total_chars = 0
    stop = False
    candidate_ranks: dict[str, tuple[int, dict]] = {}
    for q in queries:
        retrieved = rag.search(
            thread_id,
            q,
            k=max(per_query, SEMANTIC_CANDIDATES_PER_QUERY),
        )
        for rank, candidate in enumerate(retrieved):
            candidate_id = candidate["id"]
            previous = candidate_ranks.get(candidate_id)
            if previous is None or rank < previous[0]:
                candidate_ranks[candidate_id] = (rank, candidate)
        if stop:
            continue
        for r in retrieved[:per_query]:
            content_text = r["content"]
            section = (r.get("section") or "").strip().lower()
            # 参考文献/附录不是论文事实证据；过短块多为孤立标题
            if section in EVIDENCE_SKIP_SECTIONS:
                continue
            if len(content_text.strip()) < MIN_EVIDENCE_CHUNK_CHARS:
                continue
            if content_text in seen:
                continue
            if len(parts) >= max_chunks:
                stop = True
                break
            section = r["section"].title()
            content = content_text[:MAX_CHUNK_CHARS]
            page = (
                f" | 第 {r.get('page_start', 0)}-{r.get('page_end', 0)} 页"
                if r.get("page_end", 0)
                else ""
            )
            part = f"【证据 #{n} | 来源: {section}{page}】\n{content}"
            if total_chars + len(part) > MAX_EVIDENCE_CHARS:
                stop = True
                break
            seen.add(content_text)
            parts.append(part)
            total_chars += len(part)
            n += 1
    evidence = "\n\n---\n\n".join(parts)
    if include_candidates:
        candidates = [
            candidate
            for _rank, candidate in sorted(candidate_ranks.values(), key=lambda item: item[0])
        ]
        return evidence, candidates
    return evidence


# 结构化补全的章节优先级（综述等非标准论文也能覆盖到）；参考文献/附录直接跳过
SUMMARY_SECTION_TIERS = [
    {"preamble", "abstract", "introduction", "background"},
    {"method", "architecture", "approach"},
    {"evaluation", "experiment", "results"},
    {"discussion", "conclusion"},
]
# 过短的 chunk 多为孤立标题（如「Background and Main Contributions\n2.1」），不补
MIN_FALLBACK_CHUNK_CHARS = 120
# 每个章节在兜底阶段最多取几块，保证多章节覆盖而非堆在同一节
FALLBACK_PER_SECTION = 2
def build_summary_evidence(
    rag,
    thread_id: str,
    queries: list[str] | None = None,
    section_tiers: list[set[str]] | None = None,
) -> str:
    """上传摘要专用证据：先做固定语义查询，再用「按章节结构取块」兜底。

    综述/非标准结构论文用词和章节名对不上固定查询时，语义召回会稀疏，
    因此不足预算时按章节优先级 + 页序补齐核心章节代表块（跳过参考文献/附录
    与孤立标题块），确保摘要有足够、跨章节的证据。
    """
    evidence, semantic_candidates = build_evidence(
        rag,
        thread_id,
        queries or SUMMARY_QUERIES,
        max_chunks=SEMANTIC_EVIDENCE_CHUNKS,
        include_candidates=True,
    )
    section_tiers = section_tiers or SUMMARY_SECTION_TIERS

    selected_ids = set()
    candidates_structural: list[dict] = []
    if len([part for part in evidence.split("\n\n---\n\n") if part.strip()]) < MAX_EVIDENCE_CHUNKS:
        for r in rag.list_chunks(thread_id):
            section = (r.get("section") or "").strip().lower()
            if section in EVIDENCE_SKIP_SECTIONS:
                continue
            if len((r.get("content") or "").strip()) < MIN_FALLBACK_CHUNK_CHARS:
                continue
            candidates_structural.append(r)
        # build_evidence 返回的块要解析出 id 比较麻烦，这里用正文去重即可
        existing_bodies = {
            b.split("】", 1)[-1].strip()
            for b in evidence.split("\n\n---\n\n")
            if b.strip()
        }
        candidates_structural = [
            r for r in candidates_structural
            if r["content"][:MAX_CHUNK_CHARS] not in existing_bodies
        ]
        semantic_rank = {
            candidate["id"]: rank
            for rank, candidate in enumerate(semantic_candidates)
        }
        candidates_structural.sort(
            key=lambda r: (
                semantic_rank.get(r["id"], float("inf")),
                r.get("page_start", 0) or 0,
                r.get("_order", 0),
            )
        )
    else:
        existing_bodies = set()

    picked_bodies = set(existing_bodies)
    parts = [b for b in evidence.split("\n\n---\n\n") if b.strip()]
    total = sum(len(b) for b in parts)
    n = len(parts) + 1

    for tier in section_tiers:
        if len(parts) >= MAX_EVIDENCE_CHUNKS or total >= MAX_EVIDENCE_CHARS:
            break
        per_section: dict[str, int] = {}
        # 同一优先级内按页序遍历，每个章节取前 FALLBACK_PER_SECTION 块
        for r in candidates_structural:
            if len(parts) >= MAX_EVIDENCE_CHUNKS or total >= MAX_EVIDENCE_CHARS:
                break
            section = (r.get("section") or "").strip().lower()
            content = r["content"][:MAX_CHUNK_CHARS]
            if (
                section not in tier
                or r["id"] in selected_ids
                or content in picked_bodies
            ):
                continue
            if per_section.get(section, 0) >= FALLBACK_PER_SECTION:
                continue
            page = (
                f" | 第 {r.get('page_start', 0)}-{r.get('page_end', 0)} 页"
                if r.get("page_end", 0)
                else ""
            )
            part = f"【证据 #{n} | 来源: {r['section'].title()}{page}】\n{content}"
            if total + len(part) > MAX_EVIDENCE_CHARS:
                break
            selected_ids.add(r["id"])
            per_section[section] = per_section.get(section, 0) + 1
            parts.append(part)
            total += len(part)
            n += 1

    return "\n\n---\n\n".join(parts)


QUESTION_SECTION_TIERS = {
    "A1": SUMMARY_SECTION_TIERS,
    "A2": [
        {"discussion", "conclusion", "future work"},
        {"method", "architecture", "approach"},
        {"evaluation", "experiment", "results"},
        {"preamble", "abstract", "introduction", "background"},
    ],
    "C": [
        {"method", "architecture", "approach"},
        {"discussion", "conclusion", "future work"},
        {"evaluation", "experiment", "results"},
        {"preamble", "abstract", "introduction", "background"},
    ],
}


def build_question_evidence(rag, thread_id: str, route: dict) -> str:
    """Build question evidence with the same structural fallback as summaries."""
    tiers = QUESTION_SECTION_TIERS.get(route["category"], SUMMARY_SECTION_TIERS)
    return build_summary_evidence(rag, thread_id, route["queries"], tiers)


# 各类问题的「本轮回答要求」（注入系统提示，配合证据铁律与三类分层）
ROUTE_GUIDANCE = {
    "A1": (
        "请基于【本轮论文证据】回答。结论中的一级事实（贡献/方法/实验数据/结论）"
        "必须附【来源章节】+【原文引用】；证据不足的部分明确标注'论文中未检索到'。不要调用任何工具。"
    ),
    "A2": (
        "请基于【本轮论文证据】区分两类：论文明确表述的不足（附【来源】+【原文引用】）；"
        "从论文设计推导出的潜在问题（标注'论文未明确提及，属从论文设计推导'）。不要调用任何工具。"
    ),
    "C": (
        "请结合【本轮论文证据】与你的领域知识回答。引用论文处标【来源】；"
        "领域知识标注'（非论文内容，属领域背景）'；你的推断标注'（基于领域知识的推测）'。不要调用任何工具。"
    ),
}

GUIDANCE_B = (
    "该问题属于通用领域知识，与上传的论文无关，本轮不提供论文证据。"
    "请直接基于你的已有知识回答，并在回答开头显式标注："
    "'以下内容来自模型自身知识，不来自上传论文'。不要调用任何工具，也不要把通用知识伪装成论文内容。"
)

GUIDANCE_NO_EVIDENCE = (
    "该问题需要论文内容，但当前会话没有可用的论文证据（可能尚未上传论文）。"
    "请礼貌提示用户先上传论文 PDF，不要用外部知识代替，不要调用任何工具。"
)

GUIDANCE_SUMMARY = (
    "用户上传了新论文，希望生成结构化摘要。请严格按系统提示中【工作流程·场景一】的结构，"
    "仅基于【本轮论文证据】生成摘要，不要调用任何工具；证据未覆盖的章节如实标注'论文中未检索到'。"
)

GUIDANCE_SUMMARY_EMPTY = (
    "用户上传了论文，但本轮没有检索到可用证据。请明确说明暂未能读取到论文内容、无法生成摘要，"
    "不要编造论文信息，不要调用任何工具。"
)


def route_user_question(rag, thread_id: str, text: str) -> dict:
    """非上传问题：决定是否检索，返回当轮 context（evidence / guidance）。"""
    route = build_route(text)
    category = route["category"]

    if not route["needs_retrieval"]:
        return {"evidence": "", "guidance": GUIDANCE_B}

    evidence = build_question_evidence(rag, thread_id, route)
    if not evidence:
        return {"evidence": "", "guidance": GUIDANCE_NO_EVIDENCE}

    return {"evidence": evidence, "guidance": ROUTE_GUIDANCE.get(category, ROUTE_GUIDANCE["A1"])}


def prune_tool_messages(agent, config: dict) -> None:
    """删除历史里的工具消息，避免旧架构遗留的大段 ToolMessage 在后续轮次被重发。

    新架构证据走系统提示、不产生 ToolMessage；这里仅用于兼容/清理旧会话历史。
    """
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


def _status(message: str) -> str:
    return f"data: {json.dumps({'type': 'status', 'message': message}, ensure_ascii=False)}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest, fastapi_req: Request):
    request_started = perf_counter()
    rag = get_rag()

    if not request.thread_id:
        thread_id = str(uuid.uuid4())
        create_session(thread_id, "新会话")
        is_new = True
    else:
        thread_id = request.thread_id
        is_new = False

    is_pdf = request.file_type == "pdf" and bool(request.image_url)
    # 仅网络下载放在 async 里；PDF 解析/入库与检索都是 CPU/阻塞操作，
    # 移到下面的同步 SSE 生成器（由线程池执行），期间还能向前端推送进度
    download_started = perf_counter()
    pdf_data = await download_from_oss(request.image_url) if is_pdf else None
    download_ms = (perf_counter() - download_started) * 1000 if is_pdf else 0.0
    user_text = request.user_text or "请帮我分析这篇论文"
    config = {"recursion_limit": 15, "configurable": {"thread_id": thread_id}}
    metric = {
        "request_id": str(uuid.uuid4()),
        "thread_id": thread_id,
        "request_kind": "pdf_summary" if is_pdf else "question",
        "success": False,
        "download_ms": round(download_ms, 2),
        "evidence_blocks": 0,
        "evidence_chars": 0,
        "output_chars": 0,
    }

    def event_stream():
        yield f"data: {json.dumps({'type': 'thread_id', 'thread_id': thread_id})}\n\n"
        try:
            agent = get_research_agent()

            paper_title = ""
            if is_pdf:
                yield _status("正在解析论文 PDF...")
                parse_started = perf_counter()
                full_text, line_pages = extract_text_from_bytes(pdf_data)
                sections, section_pages = detect_sections(full_text, line_pages)
                chunks = chunk_sections(sections, section_pages)
                metric["parse_ms"] = round((perf_counter() - parse_started) * 1000, 2)
                if not chunks:
                    raise ValueError("No readable text chunks were extracted from the PDF.")
                paper_title = guess_title(full_text)
                index_started = perf_counter()
                clear_thread_history(thread_id)
                rag.delete_collection(thread_id)
                rag.store(thread_id, chunks, paper_title)
                metric["index_ms"] = round((perf_counter() - index_started) * 1000, 2)
                if paper_title:
                    update_session_title(thread_id, paper_title)

            if is_pdf and paper_title:
                yield _status("正在检索论文关键章节...")
                retrieval_started = perf_counter()
                evidence = build_summary_evidence(rag, thread_id)
                metric["retrieval_ms"] = round((perf_counter() - retrieval_started) * 1000, 2)
                metric["route_category"] = "SUMMARY"
                guidance = GUIDANCE_SUMMARY if evidence else GUIDANCE_SUMMARY_EMPTY
                request_context = RequestContext(evidence=evidence, guidance=guidance)
                message = HumanMessage(
                    content=f"[用户上传了新论文]《{paper_title}》\n用户说：{user_text}"
                )
            else:
                yield _status("正在分析问题...")
                routing_started = perf_counter()
                route = build_route(user_text)
                metric["routing_ms"] = round((perf_counter() - routing_started) * 1000, 2)
                metric["route_category"] = route["category"]
                if not route["needs_retrieval"]:
                    request_context = RequestContext(evidence="", guidance=GUIDANCE_B)
                else:
                    yield _status("正在检索论文内容...")
                    retrieval_started = perf_counter()
                    evidence = build_question_evidence(rag, thread_id, route)
                    metric["retrieval_ms"] = round((perf_counter() - retrieval_started) * 1000, 2)
                    if evidence:
                        guidance = ROUTE_GUIDANCE.get(route["category"], ROUTE_GUIDANCE["A1"])
                    else:
                        guidance = GUIDANCE_NO_EVIDENCE
                    request_context = RequestContext(evidence=evidence, guidance=guidance)
                message = HumanMessage(content=user_text)

            evidence_text = request_context.evidence
            metric["evidence_blocks"] = len(
                [part for part in evidence_text.split("\n\n---\n\n") if part.strip()]
            )
            metric["evidence_chars"] = len(evidence_text)
            prune_tool_messages(agent, config)

            yield _status("正在生成回答...")
            generation_started = perf_counter()
            ai_text = ""
            for chunk, _metadata in agent.stream(
                {"messages": [message]},
                stream_mode="messages",
                config=config,
                context=request_context,
            ):
                if isinstance(chunk, AIMessageChunk):
                    usage = getattr(chunk, "usage_metadata", None) or {}
                    for field in ("input_tokens", "output_tokens"):
                        value = usage.get(field)
                        if isinstance(value, int):
                            metric[field] = max(metric.get(field, 0), value)
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if text:
                        ai_text += text
                        yield f"data: {json.dumps({'type': 'content', 'content': text}, ensure_ascii=False)}\n\n"
            metric["generation_ms"] = round((perf_counter() - generation_started) * 1000, 2)
            metric["output_chars"] = len(ai_text)
            metric["success"] = True
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

            if is_new and ai_text and not paper_title:
                title = ai_text.split("\n")[0].strip()[:30]
                title = title.replace("#", "").replace("*", "").strip()
                if title:
                    update_session_title(thread_id, title)
        except Exception as exc:
            metric["error_type"] = type(exc).__name__
            raise
        finally:
            metric["total_ms"] = round((perf_counter() - request_started) * 1000, 2)
            try:
                record_request_metric(metric)
            except Exception:
                pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/research/deep")
async def deep_research_endpoint(
    payload: DeepResearchRequest,
    x_research_token: str | None = Header(default=None),
):
    """Run one explicitly confirmed, local-only, bounded research experiment."""
    try:
        guard = get_research_guard()
        # Authenticate before checking thread existence to avoid exposing local IDs.
        guard.authenticate(x_research_token)
        if payload.mode == "paper_deep" and (
            not payload.thread_id or not session_exists(payload.thread_id)
        ):
            raise HTTPException(status_code=404, detail="未找到对应论文会话")
        reservation_id = guard.begin(
            x_research_token,
            payload.confirm_external_processing,
        )
    except ResearchAccessError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    result = None
    try:
        def execute():
            retriever = (
                ChromaPaperRetriever(get_rag())
                if payload.mode == "paper_deep"
                else None
            )
            return run_deep_research(
                payload.question,
                payload.thread_id or "",
                retriever,
                AliyunResearchModel(),
                mode=payload.mode,
                trace_dir=Path(__file__).parent
                / "benchmark"
                / "experiments"
                / "private"
                / "workflow_g1_live",
            )

        result = await run_in_threadpool(execute)
    except Exception as exc:
        # No result means token usage cannot be accounted for reliably. Release the
        # reservation and expose only the error type, never credentials or prompts.
        guard.cancel(reservation_id)
        raise HTTPException(
            status_code=500,
            detail=f"深度研究运行失败：{type(exc).__name__}",
        ) from exc

    input_tokens = result.get("input_tokens")
    output_tokens = result.get("output_tokens")
    actual_tokens = (
        input_tokens + output_tokens
        if isinstance(input_tokens, int) and isinstance(output_tokens, int)
        else None
    )
    guard.finish(reservation_id, actual_tokens)
    result["local_limits"] = guard.snapshot()
    result["external_processing"] = {
        "provider": "configured Aliyun-compatible endpoint",
        "confirmed": True,
        "payload_chars": result.get("external_payload_chars", 0),
    }
    return result


@app.post("/research/adaptive")
async def adaptive_research_endpoint(
    payload: AdaptiveResearchRequest,
    x_research_token: str | None = Header(default=None),
):
    """Run auto-selected general, full-paper or bounded-RAG research."""
    try:
        guard = get_research_guard()
        guard.authenticate(x_research_token)
        if payload.thread_id and not session_exists(payload.thread_id):
            raise HTTPException(status_code=404, detail="未找到对应论文会话")
        reservation_id = guard.begin(
            x_research_token,
            payload.confirm_external_processing,
        )
    except ResearchAccessError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    result = None
    try:
        def execute_adaptive():
            rag = get_rag() if payload.thread_id else None
            rows = rag.list_chunks(payload.thread_id) if rag is not None else None
            retriever = ChromaPaperRetriever(rag) if rag is not None else None
            conversation = (
                list_adaptive_turns(payload.thread_id, limit=4)
                if payload.thread_id and payload.history_mode == "continue" else []
            )
            try:
                full_context_chars = int(
                    os.getenv("ADAPTIVE_FULL_CONTEXT_MAX_CHARS", "100000")
                )
            except ValueError as exc:
                raise RuntimeError("Invalid ADAPTIVE_FULL_CONTEXT_MAX_CHARS") from exc
            return run_adaptive_research(
                payload.question,
                payload.thread_id,
                rows,
                retriever,
                AliyunResearchModel(),
                policy=AdaptivePolicy(full_context_max_chars=full_context_chars),
                strategy=payload.strategy,
                answer_depth=payload.answer_depth,
                conversation_history=conversation,
                history_mode=payload.history_mode,
                explanation_pipeline=payload.explanation_pipeline,
                enable_quality_review=(payload.enable_quality_review if payload.enable_quality_review is not None
                                       else os.getenv("RESEARCH_QUALITY_REVIEW", "1").lower() in {"1", "true", "yes"}),
                trace_dir=ADAPTIVE_TRACE_DIR,
            )

        result = await run_in_threadpool(execute_adaptive)
    except ValueError as exc:
        guard.cancel(reservation_id)
        raise HTTPException(status_code=400, detail=f"研究配置不适用：{exc}") from exc
    except Exception as exc:
        guard.cancel(reservation_id)
        raise HTTPException(
            status_code=500,
            detail=f"自适应研究运行失败：{type(exc).__name__}",
        ) from exc

    input_tokens = result.get("input_tokens")
    output_tokens = result.get("output_tokens")
    actual_tokens = (
        input_tokens + output_tokens
        if isinstance(input_tokens, int) and isinstance(output_tokens, int)
        else None
    )
    guard.finish(reservation_id, actual_tokens)
    if payload.thread_id and result.get("status") in {
        "ok", "ok_with_limits", "insufficient", "partial"
    }:
        try:
            record_adaptive_turn({
                "request_id": result["request_id"],
                "thread_id": payload.thread_id,
                "question": payload.question,
                "answer": result.get("answer") or "",
                "strategy": payload.strategy,
                "answer_depth": payload.answer_depth,
                "mode": result.get("mode") or "unknown",
                "status": result["status"],
            })
        except Exception:
            # A paid answer should still reach the user if optional conversation
            # persistence fails. The administrator can diagnose the exception.
            logger.exception(
                "Failed to persist adaptive conversation turn for request %s",
                result.get("request_id"),
            )
    # Operational metrics remain in the private trace and are intentionally
    # not returned to the end-user response.
    return public_research_result(result)


def _authenticate_trace_admin(token: str | None) -> None:
    try:
        get_research_guard().authenticate(token)
    except ResearchAccessError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.get("/admin/research")
async def research_trace_page():
    """Serve the shell only; trace data still requires the admin header token."""
    return FileResponse(Path(__file__).parent / "static" / "admin-traces.html")


@app.get("/research")
async def adaptive_research_page():
    """Local pilot UI; the access token stays in page memory only."""
    return FileResponse(Path(__file__).parent / "static" / "research.html")


@app.get("/admin/research/traces")
async def list_research_traces(
    limit: int = 50,
    x_research_token: str | None = Header(default=None),
):
    _authenticate_trace_admin(x_research_token)
    return {"items": TraceRepository(ADAPTIVE_TRACE_DIR).list(limit=max(1, min(limit, 200)))}


@app.get("/admin/research/traces/{request_id}")
async def get_research_trace(
    request_id: str,
    x_research_token: str | None = Header(default=None),
):
    _authenticate_trace_admin(x_research_token)
    repository = TraceRepository(ADAPTIVE_TRACE_DIR)
    try:
        return repository.get(request_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="未找到该运行记录") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="运行记录编号无效") from exc


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
async def refresh(thread_id: str) -> list[dict]:
    agent = get_research_agent()
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
