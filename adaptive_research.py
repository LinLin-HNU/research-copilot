"""Deterministic mode selection and public response projection.

The language model never decides whether it receives a full paper or RAG
evidence.  The server chooses from explicit source count and character budget.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from quality_review import add_usage, cache_total, review_once
from reading_limits import FULL_PAPER_OUTPUT_LIMITS


@dataclass(frozen=True)
class AdaptivePolicy:
    """Limits used before any model call is made."""

    # The local corpus is English-heavy PDF text. This conservative character
    # guard is based on the observed project payloads; it deliberately does
    # not pretend that character count is an exact tokenizer or provider limit.
    full_context_max_chars: int = 100_000
    full_context_max_chunks: int = 200
    estimated_rag_input_tokens: int = 6_000
    full_context_chars_per_token: float = 2.5
    full_context_prompt_reserve_tokens: int = 1_500
    max_auto_full_to_rag_ratio: float = 6.0

    def __post_init__(self) -> None:
        if (self.full_context_max_chars <= 0 or self.full_context_max_chunks <= 0
                or self.estimated_rag_input_tokens <= 0
                or self.full_context_chars_per_token <= 0
                or self.full_context_prompt_reserve_tokens <= 0
                or self.max_auto_full_to_rag_ratio <= 0):
            raise ValueError("Adaptive policy limits must be positive")


@dataclass(frozen=True)
class ModeDecision:
    mode: str
    reason: str
    paper_count: int
    paper_chars: int
    paper_chunks: int
    strategy: str = "deep"
    question_scope: str = "not_applicable"
    estimated_full_input_tokens: int = 0
    estimated_rag_input_tokens: int = 0
    estimated_cost_ratio: float = 0.0
    full_context_eligible: bool = False


GLOBAL_SCOPE_MARKERS = (
    "总结", "概述", "综述", "整体", "全文", "主要贡献", "核心贡献", "全部局限",
    "优缺点", "完整架构", "系统比较", "全面比较", "全面讲解", "整体讲解",
    "讲解这篇论文", "讲解一下这篇论文", "从头讲", "通读", "overall", "overview",
    "summarize", "summary", "main contribution", "limitations", "entire paper",
)


def classify_question_scope(question: str) -> str:
    normalized = re.sub(r"\s+", " ", (question or "").strip().lower())
    return "global" if any(marker in normalized for marker in GLOBAL_SCOPE_MARKERS) else "local"


def _paper_stats(papers: Iterable[list[dict]]) -> tuple[list[list[dict]], int, int]:
    materialized = list(papers)
    total_chars = 0
    total_chunks = 0
    for rows in materialized:
        if not isinstance(rows, list):
            raise ValueError("invalid_paper_rows")
        total_chunks += len(rows)
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("content"), str):
                raise ValueError("invalid_paper_chunk")
            total_chars += len(row["content"])
    return materialized, total_chars, total_chunks


def select_research_mode(
    papers: Iterable[list[dict]], *, question: str = "", strategy: str = "deep",
    policy: AdaptivePolicy | None = None,
) -> ModeDecision:
    """Select from source size, coarse question scope and explicit user strategy."""

    policy = policy or AdaptivePolicy()
    if strategy not in {"auto", "fast", "deep"}:
        raise ValueError("unsupported_strategy")
    materialized, total_chars, total_chunks = _paper_stats(papers)
    paper_count = len(materialized)
    scope = classify_question_scope(question) if paper_count else "not_applicable"
    estimated_full = math.ceil(
        (total_chars + total_chunks * 80) / policy.full_context_chars_per_token
    ) + policy.full_context_prompt_reserve_tokens if paper_count else 0
    ratio = (
        estimated_full / policy.estimated_rag_input_tokens
        if estimated_full else 0.0
    )
    eligible = (
        paper_count == 1
        and total_chars <= policy.full_context_max_chars
        and total_chunks <= policy.full_context_max_chunks
    )

    def decision(mode: str, reason: str) -> ModeDecision:
        return ModeDecision(
            mode, reason, paper_count, total_chars, total_chunks,
            strategy=strategy, question_scope=scope,
            estimated_full_input_tokens=estimated_full,
            estimated_rag_input_tokens=policy.estimated_rag_input_tokens,
            estimated_cost_ratio=round(ratio, 2),
            full_context_eligible=eligible,
        )

    if paper_count == 0:
        return decision("general", "no_paper_selected")
    if paper_count > 1:
        return decision("paper_deep", "multiple_papers_selected")
    if total_chars > policy.full_context_max_chars:
        return decision("paper_deep", "paper_exceeds_full_context_budget")
    if total_chunks > policy.full_context_max_chunks:
        return decision("paper_deep", "paper_exceeds_full_context_chunk_limit")
    if strategy == "fast":
        return decision("paper_deep", "user_selected_fast")
    if strategy == "deep":
        return decision("paper_full", "user_selected_deep")
    if scope == "global" and ratio <= policy.max_auto_full_to_rag_ratio:
        return decision("paper_full", "auto_global_question_within_cost_guard")
    if scope == "global":
        return decision("paper_deep", "auto_global_question_exceeds_cost_guard")
    return decision("paper_deep", "auto_local_question_prefers_rag")


def build_full_paper_context(rows: list[dict], *, max_chars: int) -> tuple[str, list[dict]]:
    """Build complete, citation-addressable context or fail without truncation."""

    if max_chars <= 0:
        raise ValueError("full_context_budget_exceeded")
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("page_start", 0) or 0,
            row.get("_order", 0) or 0,
            str(row.get("id") or ""),
        ),
    )
    if sum(len(row.get("content", "")) for row in ordered) > max_chars:
        raise ValueError("full_context_budget_exceeded")

    evidence: list[dict[str, Any]] = []
    labels: list[str] = []
    seen_ids: set[str] = set()
    for row in ordered:
        chunk_id = row.get("id")
        content = row.get("content")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen_ids:
            raise ValueError("invalid_or_duplicate_chunk_id")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("invalid_paper_chunk")
        section = str(row.get("section") or "unknown")[:120]
        page_start = row.get("page_start") or 0
        page_end = row.get("page_end") or page_start
        if not isinstance(page_start, int) or not isinstance(page_end, int):
            raise ValueError("invalid_page_metadata")
        item = {
            "id": chunk_id,
            "section": section,
            "page_start": page_start,
            "page_end": page_end,
            "text": content,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }
        evidence.append(item)
        labels.append(f"[{chunk_id} | {section} | p.{page_start}-{page_end}]\n{content}")
        seen_ids.add(chunk_id)
    return "\n\n".join(labels), evidence


PUBLIC_RESULT_FIELDS = (
    "request_id",
    "mode",
    "status",
    "answer",
    "sources",
    "unresolved",
    "caveats",
    "suggestion",
)


def public_research_result(result: dict) -> dict:
    """Return only fields useful to an end user; operational data stays private."""

    public = {name: result.get(name) for name in PUBLIC_RESULT_FIELDS}
    for name in ("sources", "unresolved", "caveats"):
        if public[name] is None:
            public[name] = []
    if public["status"] in {"stopped", "error"}:
        public["answer"] = "本次分析未能完成，暂时没有可核验的论文结论。"
        public["suggestion"] = {
            "provider_quota_exhausted": "模型服务额度不足，请联系管理员补充额度或切换可用模型后再试。",
            "provider_request_rejected": "模型服务拒绝了请求，请联系管理员检查模型权限、密钥或调用限制。",
            "output_token_limit": "本次讲解输出达到长度上限，请缩小问题范围或联系管理员检查输出预算。",
        }.get(result.get("stop_reason"), "请重试；如果再次出现，请把请求编号交给管理员查看运行记录。")
    elif result.get("stop_reason") == "output_token_limit_partial":
        public["suggestion"] = "已保留部分回答；可针对未展开的部分继续追问，无需重复请求整篇讲解。"
    return public


def _record_mode_selection(trace_dir: Path | None, result: dict, decision: ModeDecision,
                           request_context: dict | None = None) -> None:
    if trace_dir is None:
        return
    target = Path(trace_dir) / f"{result['request_id']}.json"
    if not target.is_file():
        return
    try:
        trace = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    trace["mode"] = decision.mode
    trace["request_context"] = request_context
    trace["mode_reason"] = decision.reason
    trace["strategy"] = decision.strategy
    trace["question_scope"] = decision.question_scope
    trace["deep_retry_recommended"] = bool(result.get("deep_retry_recommended"))
    trace["events"] = [{
        "step": "mode_selection",
        "mode": decision.mode,
        "reason": decision.reason,
        "paper_count": decision.paper_count,
        "paper_chars": decision.paper_chars,
        "paper_chunks": decision.paper_chunks,
        "strategy": decision.strategy,
        "question_scope": decision.question_scope,
        "estimated_full_input_tokens": decision.estimated_full_input_tokens,
        "estimated_rag_input_tokens": decision.estimated_rag_input_tokens,
        "estimated_cost_ratio": decision.estimated_cost_ratio,
    }] + list(trace.get("events") or [])
    if result.get("deep_retry_recommended"):
        trace["events"].append({
            "step": "recommendation",
            "action": "manual_deep_retry",
            "reason": "model_reported_unresolved_items",
        })
    target.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")


def run_adaptive_research(
    question: str,
    thread_id: str,
    paper_rows: list[dict] | None,
    retriever: Any,
    model: Any,
    *,
    policy: AdaptivePolicy | None = None,
    rag_budget: Any = None,
    strategy: str = "auto",
    answer_depth: str = "beginner",
    conversation_history: list[dict] | None = None,
    history_mode: str = "continue",
    explanation_pipeline: str = "direct",
    allow_full_fallback: bool = False,
    enable_quality_review: bool = False,
    trace_dir: Path | None = None,
    clock=time.monotonic,
) -> dict:
    """Execute one adaptively selected request with a shared result contract."""

    from deep_research import (
        WorkflowStop,
        _build_public_sources,
        _normalize_final_contract,
        _render_final,
        _save_private_trace,
        run_deep_research,
    )

    policy = policy or AdaptivePolicy()
    if answer_depth not in {"beginner", "concise", "advanced"}:
        raise ValueError("unsupported_answer_depth")
    if history_mode not in {"fresh", "continue"}:
        raise ValueError("unsupported_history_mode")
    if explanation_pipeline not in {"direct", "structured"}:
        raise ValueError("unsupported_explanation_pipeline")
    conversation_history = [] if history_mode == "fresh" else list(conversation_history or [])[-4:]
    request_context = {
        "history_mode": history_mode,
        "history_request_ids": [row.get("request_id") for row in conversation_history],
        "history_sha256": hashlib.sha256(json.dumps(conversation_history, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
        "paper_sha256": hashlib.sha256(json.dumps(sorted(
            [(row.get("id", ""), row.get("content", "")) for row in paper_rows or []]),
            ensure_ascii=False).encode("utf-8")).hexdigest(),
        "answer_depth": answer_depth,
        "explanation_pipeline": explanation_pipeline,
    }
    papers = [paper_rows] if paper_rows else []
    decision = select_research_mode(
        papers, question=question, strategy=strategy, policy=policy
    )
    if explanation_pipeline == "structured" and (decision.mode != "paper_full" or answer_depth != "advanced"):
        raise ValueError("structured_requires_full_paper_and_advanced")

    if decision.mode in {"general", "paper_deep"}:
        result = run_deep_research(
            question,
            thread_id,
            retriever,
            model,
            mode="general" if decision.mode == "general" else "paper_deep",
            budget=rag_budget,
            answer_depth=answer_depth,
            conversation_history=conversation_history,
            enable_quality_review=enable_quality_review,
            trace_dir=trace_dir,
            clock=clock,
        )
        result["mode"] = decision.mode
        result["mode_reason"] = decision.reason
        result["strategy"] = strategy
        result["answer_depth"] = answer_depth
        result["deep_retry_recommended"] = bool(
            decision.mode == "paper_deep"
            and strategy == "auto"
            and decision.full_context_eligible
            and result.get("status") in {"insufficient", "partial"}
        )
        result["suggestion"] = (
            "当前答案被模型标记为仍有未解决项；如需全文复核，可手动选择 deep。"
            if result["deep_retry_recommended"] else None
        )
        _record_mode_selection(trace_dir, result, decision, request_context)
        if (decision.mode == "paper_deep" and strategy == "auto"
                and allow_full_fallback and decision.full_context_eligible
                and result.get("status") in {"insufficient", "partial"}):
            fallback = run_adaptive_research(
                question, thread_id, paper_rows, retriever, model,
                policy=policy, rag_budget=rag_budget, strategy="deep",
                answer_depth=answer_depth,
                conversation_history=conversation_history,
                history_mode=history_mode,
                explanation_pipeline=explanation_pipeline,
                allow_full_fallback=False, trace_dir=trace_dir, clock=clock,
                enable_quality_review=enable_quality_review,
            )
            prior = dict(result)
            for name in ("model_calls", "actions", "retrieval_calls", "evidence_chars",
                         "external_payload_chars", "elapsed_ms"):
                fallback[name] = (fallback.get(name) or 0) + (prior.get(name) or 0)
            fallback["cached_tokens"] = (
                fallback["cached_tokens"] + prior["cached_tokens"]
                if isinstance(fallback.get("cached_tokens"), int)
                and isinstance(prior.get("cached_tokens"), int) else None
            )
            fallback["evidence_ids"] = sorted(
                set(fallback.get("evidence_ids") or [])
                | set(prior.get("evidence_ids") or [])
            )
            fallback["unique_evidence_ids"] = len(fallback["evidence_ids"])
            for name in ("input_tokens", "output_tokens"):
                if fallback.get(name) is None or prior.get(name) is None:
                    fallback[name] = None
                else:
                    fallback[name] += prior[name]
            fallback["fallback_used"] = True
            fallback["fallback_from_request_id"] = prior.get("request_id")
            fallback["fallback_reason"] = prior.get("stop_reason")
            fallback["strategy"] = "auto"
            fallback["mode_reason"] = "auto_rag_partial_fallback_to_full"
            _merge_fallback_trace(trace_dir, fallback, prior)
            _mark_trace_superseded(trace_dir, prior.get("request_id"), fallback["request_id"])
            return fallback
        result["fallback_used"] = False
        return result

    started = clock()
    request_id = str(uuid.uuid4())
    context, evidence = build_full_paper_context(
        paper_rows or [], max_chars=(min(policy.full_context_max_chars, 100_000)
                                    if explanation_pipeline == "structured" else policy.full_context_max_chars)
    )
    trace: dict[str, Any] = {
        "request_context": request_context,
        "request_id": request_id,
        "thread_id": thread_id,
        "mode": decision.mode,
        "mode_reason": decision.reason,
        "strategy": strategy,
        "answer_depth": answer_depth,
        "question_scope": decision.question_scope,
        "question": question,
        "budget": asdict(policy),
        "events": [{
            "step": "mode_selection",
            "mode": decision.mode,
            "reason": decision.reason,
            "paper_count": decision.paper_count,
            "paper_chars": decision.paper_chars,
            "paper_chunks": decision.paper_chunks,
            "strategy": decision.strategy,
            "question_scope": decision.question_scope,
            "estimated_full_input_tokens": decision.estimated_full_input_tokens,
            "estimated_rag_input_tokens": decision.estimated_rag_input_tokens,
            "estimated_cost_ratio": decision.estimated_cost_ratio,
        }, {
            "step": "full_context",
            "provided_segments": [{
                "id": item["id"],
                "section": item["section"],
                "page_start": item["page_start"],
                "page_end": item["page_end"],
                "sha256": item["sha256"],
            } for item in evidence],
            "evidence_chars": len(context),
        }],
        "model_calls": 0,
        "retrieval_calls": 0,
        "actions": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_tokens": 0,
        "cache_usage_missing_calls": 0,
        "usage_missing_calls": 0,
        "external_payload_chars": 0,
        "explanation_pipeline": explanation_pipeline,
        "generation_limits": {"max_calls": 2 if explanation_pipeline == "structured" else 1,
                              "max_input_chars_per_call": 120000, "max_total_input_chars": 240000,
                              "deadline_seconds": 120, "max_retries": 0},
    }
    status = "stopped"
    stop_reason = "not_started"
    answer = "本次研究未能安全完成，请查看停止原因；当前没有可核验的论文结论。"
    claims: list[dict] = []
    unresolved: list[str] = []
    coverage: list[dict] = []
    caveats: list[str] = []
    sources: list[dict] = []

    def full_call(stage, payload, output_limit):
        from deep_research_runtime import AliyunResearchModel
        remaining_seconds = 120 - (clock() - started)
        if remaining_seconds <= 0:
            raise WorkflowStop("deadline")
        if trace["model_calls"] >= trace["generation_limits"]["max_calls"]:
            raise WorkflowStop("model_call_budget")
        timeout = min(60.0, remaining_seconds)
        describe = getattr(model, "describe_call", None) or AliyunResearchModel.describe_request
        meta = describe(stage, payload, max_output_tokens=output_limit, timeout_seconds=timeout)
        size = meta["serialized_input_chars"]
        if size > 120000 or trace["external_payload_chars"] + size > 240000:
            raise WorkflowStop("serialized_input_budget")
        event = {"step": "model", "stage": stage, "request_meta": meta}
        trace["events"].append(event)
        trace["model_calls"] += 1
        trace["external_payload_chars"] += size
        try:
            response = model.call(stage, payload, max_output_tokens=output_limit, timeout_seconds=timeout)
        except Exception as exc:
            add_usage(trace, {})
            event.update(error_type=type(exc).__name__, parse_error="model_call_failed")
            failure = AliyunResearchModel.failure_metadata(exc)
            event.update(failure)
            if (failure.get("provider_error_code") or "").lower() in {
                    "insufficient_quota", "quota_exceeded", "insufficient_balance", "arrearage"}:
                raise WorkflowStop("provider_quota_exhausted") from exc
            if failure.get("http_status") in {401, 403, 429}:
                raise WorkflowStop("provider_request_rejected") from exc
            raise WorkflowStop("model_call_failed") from exc
        if not isinstance(response, dict):
            add_usage(trace, {})
            event["parse_error"] = "invalid_model_response"
            raise WorkflowStop("invalid_model_response")
        usage = response.get("usage") or {}
        add_usage(trace, usage)
        event.update(raw=response.get("raw"), data=response.get("data"), usage=usage,
                     parse_error=response.get("parse_error"), finish_reason=response.get("finish_reason"),
                     output_budget_exceeded=response.get("output_budget_exceeded"),
                     output_limit_reached=response.get("output_limit_reached"))
        if clock() - started > 120:
            if (stage in {"full_context", "structured_context"}
                    and not response.get("output_budget_exceeded")
                    and (response.get("data") is not None
                         or response.get("parse_error") == "output_token_limit")):
                # The paid final response has already arrived. Validate locally,
                # keep its usable records, and do not start another model call.
                trace["late_final_response"] = True
                trace["events"].append({"step": "late_final_response", "extra_model_calls": 0})
            else:
                raise WorkflowStop("deadline")
        if response.get("parse_error") or response.get("data") is None:
            if stage in {"full_context", "structured_context"}:
                from partial_output import recover_final
                recovered, recovery = recover_final(response, {row["id"]: row for row in evidence})
                if recovery is not None:
                    trace["output_recovery"] = recovery
                    trace["events"].append({"step": "output_recovery", **recovery})
                if recovered is not None:
                    return recovered
            raise WorkflowStop(str(response.get("parse_error") or "invalid_model_response"))
        return response["data"]

    try:
        brief = None
        if explanation_pipeline == "structured":
            from paper_structure import validate_structure, compact_source_context
            compact_context, aliases = compact_source_context(evidence)
            trace["structure_source_aliases"] = aliases
            try:
                extracted = full_call("source_structure", {"paper_context": compact_context,
                                      "question": question, "answer_depth": answer_depth}, 2200)
                brief, checks = validate_structure(extracted, evidence, aliases)
                trace["structure_checks"] = checks
                trace["reading_brief_candidate"] = brief
                if brief is not None and checks.get("usable_for_structured_writer") is False:
                    trace["structure_fallback_reason"] = "incomplete_core_structure"
                    brief = None
                trace["reading_brief"] = brief
                if brief is None:
                    trace.setdefault("structure_fallback_reason", checks["reason"])
            except WorkflowStop as exc:
                trace["structure_fallback_reason"] = exc.reason
                if exc.reason in {"provider_quota_exhausted", "provider_request_rejected",
                                   "provider_output_budget_exceeded"}:
                    raise  # Authentication/quota/rate rejection cannot benefit from a writer call.
            trace["events"].append({"step": "structure_validation", "brief": brief,
                                    "checks": trace.get("structure_checks"),
                                    "fallback_reason": trace.get("structure_fallback_reason")})
        payload = {"paper_context": context, "conversation": conversation_history,
                   "question": question, "answer_depth": answer_depth}
        if brief is not None:
            payload["reading_brief"] = brief
        data = full_call("structured_context" if brief is not None else "full_context", payload,
                         FULL_PAPER_OUTPUT_LIMITS[answer_depth])
        seen = {item["id"]: item for item in evidence}
        from deep_research import _prepare_final_contract
        final, coverage, caveats, structured = _prepare_final_contract(
            data, [{"id": "q1", "text": question}], seen, True, trace
        )
        answer, claims, unresolved = _render_final(final, seen, True)
        sources = _build_public_sources(claims, seen)
        if trace.get("output_recovery", {}).get("retained_items", 0):
            from partial_output import TRUNCATION_NOTICE
            unresolved = [TRUNCATION_NOTICE]
            answer += "\n\n### 本次输出未完成\n- " + TRUNCATION_NOTICE
            status, stop_reason = "partial", "output_token_limit_partial"
        elif trace.get("late_final_response"):
            status, stop_reason = "partial", "deadline_answer_retained"
            answer += "\n\n本次响应超过总时限；已保留通过结构与来源身份校验的内容，不再追加分析。"
        elif trace.get("answer_degradation"):
            status, stop_reason = "partial", "answer_validation_partial"
            answer += "\n\n部分输出格式已调整或不合规内容已省略；保留内容通过来源身份校验，但仍需核查事实。"
        elif unresolved:
            status = "insufficient" if structured else "partial"
            stop_reason = "full_context_incomplete"
        elif structured and caveats:
            status, stop_reason = "ok_with_limits", "answer_with_caveats"
        else:
            status, stop_reason = "ok", "full_context_answer"
        trace["events"].append({
            "step": "finish", "stop_reason": stop_reason,
            "claims": claims, "unresolved": unresolved,
            "coverage": coverage, "caveats": caveats,
        })
    except WorkflowStop as exc:
        stop_reason = exc.reason
        unresolved = [stop_reason]
        trace["events"].append({"step": "finish", "stop_reason": stop_reason})
    except Exception as exc:
        status, stop_reason = "error", "component_error"
        unresolved = [stop_reason]
        trace["events"].append({
            "step": "finish", "stop_reason": stop_reason, "error_type": type(exc).__name__,
        })

    quality_report = review_once(
        model, question, answer, evidence, trace,
        enabled=enable_quality_review and explanation_pipeline != "structured"
        and not trace.get("output_recovery") and not trace.get("answer_degradation")
        and not trace.get("late_final_response")
        and status in {"ok", "ok_with_limits", "partial", "insufficient"},
    )
    trace["cached_tokens"] = cache_total(trace)
    brief_used = bool(trace.get("reading_brief")) and any(
        event.get("stage") == "structured_context" for event in trace["events"])
    trace.update({
        "reading_brief_used": brief_used,
        "reading_brief_completeness": (trace.get("structure_checks") or {}).get("completeness"),
        "quality_review": quality_report,
        "status": status,
        "stop_reason": stop_reason,
        "answer": answer,
        "claims": claims,
        "unresolved": unresolved,
        "coverage": coverage,
        "caveats": caveats,
        "sources": sources,
        "evidence_chars": len(context),
        "unique_evidence_ids": len(evidence),
        "evidence_ids": [item["id"] for item in evidence],
        "elapsed_ms": round((clock() - started) * 1000, 2),
    })
    _save_private_trace(trace, trace_dir)
    return {
        "request_id": request_id,
        "mode": decision.mode,
        "mode_reason": decision.reason,
        "strategy": strategy,
        "answer_depth": answer_depth,
        "explanation_pipeline": explanation_pipeline,
        "reading_brief_used": brief_used,
        "reading_brief_completeness": (trace.get("structure_checks") or {}).get("completeness"),
        "structure_fallback_reason": trace.get("structure_fallback_reason"),
        "status": status,
        "stop_reason": stop_reason,
        "answer": answer,
        "claims": claims,
        "unresolved": unresolved,
        "coverage": coverage,
        "caveats": caveats,
        "sources": sources,
        "model_calls": trace["model_calls"],
        "actions": 0,
        "retrieval_calls": 0,
        "evidence_chars": len(context),
        "unique_evidence_ids": len(evidence),
        "input_tokens": trace["input_tokens"] if not trace["usage_missing_calls"] else None,
        "output_tokens": trace["output_tokens"] if not trace["usage_missing_calls"] else None,
        "cached_tokens": trace["cached_tokens"],
        "usage_missing_calls": trace["usage_missing_calls"],
        "external_payload_chars": trace["external_payload_chars"],
        "elapsed_ms": trace["elapsed_ms"],
        "steps": [
            {"step": event["step"], "reason": event.get("reason"), "action": event.get("action")}
            for event in trace["events"]
        ],
        "fallback_used": False,
    }


def _mark_trace_superseded(trace_dir: Path | None, request_id: str | None,
                           replacement_id: str) -> None:
    if trace_dir is None or not request_id:
        return
    target = Path(trace_dir) / f"{request_id}.json"
    if not target.is_file():
        return
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        data["superseded_by"] = replacement_id
        target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError):
        return


def _merge_fallback_trace(trace_dir: Path | None, fallback: dict, prior: dict) -> None:
    if trace_dir is None:
        return
    target = Path(trace_dir) / f"{fallback['request_id']}.json"
    if not target.is_file():
        return
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    data["events"] = [{
        "step": "fallback",
        "from_request_id": prior.get("request_id"),
        "from_mode": prior.get("mode", "paper_deep"),
        "reason": prior.get("stop_reason"),
        "prior_model_calls": prior.get("model_calls"),
        "prior_retrieval_calls": prior.get("retrieval_calls"),
    }] + list(data.get("events") or [])
    for name in ("model_calls", "actions", "retrieval_calls", "evidence_chars",
                 "unique_evidence_ids", "external_payload_chars", "elapsed_ms",
                 "cached_tokens", "input_tokens", "output_tokens"):
        data[name] = fallback.get(name)
    data["fallback_used"] = True
    data["fallback_from_request_id"] = prior.get("request_id")
    data["fallback_reason"] = prior.get("stop_reason")
    data["strategy"] = fallback.get("strategy")
    data["mode_reason"] = fallback.get("mode_reason")
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
