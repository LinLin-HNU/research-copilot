"""Read-only, local trace presentation helpers for administrators."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


TRACE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
STEP_LABELS = {
    "mode_selection": "模式选择",
    "plan": "规划",
    "retrieval": "首次检索",
    "model_proposal": "模型建议",
    "assessment": "证据判断",
    "gate": "动作门禁",
    "followup": "补充检索",
    "full_context": "整篇上下文",
    "model": "模型调用",
    "postcheck": "生成后检查",
    "finish": "完成",
    "fallback": "全文兜底",
    "recommendation": "后续建议",
    "quality_review": "关键事实辅助核验",
    "structure_validation": "原文资料表检查",
}


def _event_result(event: dict) -> str:
    step = event.get("step")
    if step == "gate":
        prefix = "放行" if event.get("allowed") else "拒绝"
        return f"{prefix}：{event.get('reason') or '未说明'}"
    if step in {"retrieval", "followup"}:
        provided = event.get("provided_segments") or []
        return f"提供 {len(provided)} 个证据片段"
    if step == "assessment":
        action = event.get("action") or {}
        return f"建议动作：{action.get('type') or '未知'}"
    if step == "model":
        return f"阶段：{event.get('stage') or '未知'}"
    if step == "finish":
        return f"停止原因：{event.get('stop_reason') or '未知'}"
    if step == "quality_review":
        return f"{event.get('status', '未知')}：{len(event.get('issues') or [])} 处待复核疑点；未改写答案"
    if step == "structure_validation":
        return (f"降级：{event['fallback_reason']}" if event.get("fallback_reason") else
                "已检查资料表格式和引用；语义仍需复核")
    return "已记录"


def summarize_trace(trace: dict) -> dict:
    input_tokens = trace.get("input_tokens")
    output_tokens = trace.get("output_tokens")
    total_tokens = (
        input_tokens + output_tokens
        if isinstance(input_tokens, int) and isinstance(output_tokens, int)
        else None
    )
    timeline = []
    for index, event in enumerate(trace.get("events") or [], start=1):
        if not isinstance(event, dict):
            continue
        step = str(event.get("step") or "unknown")
        timeline.append({
            "index": index,
            "step": step,
            "label": STEP_LABELS.get(step, step),
            "result": _event_result(event),
            "detail": event,
        })
    return {
        "request_id": trace.get("request_id"),
        "thread_id": trace.get("thread_id"),
        "question": trace.get("question"),
        "mode": trace.get("mode"),
        "status": trace.get("status"),
        "stop_reason": trace.get("stop_reason"),
        "decision": {
            "strategy": trace.get("strategy"),
            "question_scope": trace.get("question_scope"),
            "mode_reason": trace.get("mode_reason"),
            "deep_retry_recommended": trace.get("deep_retry_recommended", False),
        },
        "usage": {
            "model_calls": trace.get("model_calls", 0),
            "retrieval_calls": trace.get("retrieval_calls", 0),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": trace.get("cached_tokens"),
            "cache_usage_missing_calls": trace.get("cache_usage_missing_calls"),
        },
        "evidence": {
            "chars": trace.get("evidence_chars", 0),
            "unique_ids": trace.get("unique_evidence_ids", 0),
        },
        "timing": {"elapsed_ms": trace.get("elapsed_ms")},
        "quality_review": trace.get("quality_review"),
        "request_context": trace.get("request_context"),
        "explanation_pipeline": trace.get("explanation_pipeline", "direct"),
        "reading_brief": trace.get("reading_brief"),
        "structure_checks": trace.get("structure_checks"),
        "timeline": timeline,
    }


class TraceRepository:
    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid_trace") from exc
        if not isinstance(value, dict):
            raise ValueError("invalid_trace")
        return value

    def list(self, limit: int = 50) -> list[dict]:
        if limit <= 0:
            return []
        if not self.root.exists():
            return []
        paths = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        results = []
        for path in paths:
            try:
                raw = self._load(path)
                if raw.get("superseded_by"):
                    continue
                summary = summarize_trace(raw)
                # The list endpoint stays compact. Full per-event details are
                # loaded only after an administrator selects one request.
                summary["timeline"] = [
                    {name: item[name] for name in ("index", "step", "label", "result")}
                    for item in summary["timeline"]
                ]
                results.append(summary)
            except ValueError:
                continue
            if len(results) >= min(limit, 200):
                break
        return results

    def get(self, request_id: str) -> dict:
        if not isinstance(request_id, str) or not TRACE_ID.fullmatch(request_id):
            raise ValueError("invalid_request_id")
        target = self.root / f"{request_id}.json"
        if not target.is_file():
            raise FileNotFoundError(request_id)
        return summarize_trace(self._load(target))
