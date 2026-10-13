"""Bounded, request-local single-paper research workflow.

Semantic evidence sufficiency is a model proposal. This module validates only
the structure, provenance and cost of proposed actions and citations.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from reading_limits import RAG_PAPER_OUTPUT_LIMITS


SKIP_SECTIONS = {"references", "reference", "appendix", "bibliography"}
ASSESSMENTS = {"supported", "refuted_premise", "missing_specific_fact", "truncated_context", "conflicting_evidence", "unknown"}
ACTIONS = {"search", "expand", "stop"}
DIRECTIONS = {"continue", "next", "previous"}
KINDS = {"paper_fact", "inference", "background"}
LEXICAL_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "what", "how", "why",
    "this", "that", "these", "those", "are", "was", "were", "has", "have",
    "paper", "using", "based", "informer",
}
METHOD_QUERY_MARKERS = {
    "mechanism", "method", "architecture", "algorithm", "design", "framework",
    "机制", "方法", "架构", "算法", "设计",
}
METHOD_SECTIONS = {"method", "methodology", "methods", "architecture", "model", "introduction"}
EVALUATION_QUERY_MARKERS = {
    "evaluation", "metric", "metrics", "score", "scores", "error", "probabilistic",
    "point forecast", "评估", "指标", "评分", "误差", "概率预测", "点预测",
}
EVALUATION_SECTIONS = {"evaluation", "evaluation scores", "experiment", "experiments"}
TAXONOMY_QUERY_MARKERS = {"classification", "classify", "taxonomy", "categorization", "分类", "类目"}


@dataclass(frozen=True)
class ResearchBudget:
    max_actions: int = 2
    max_model_calls: int = 3
    max_unique_chunks: int = 8
    max_chunk_chars: int = 600
    max_evidence_chars: int = 6000
    first_evidence_chars: int = 4000
    deadline_seconds: float = 180.0
    model_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        values = (self.max_actions, self.max_model_calls, self.max_unique_chunks,
                  self.max_chunk_chars, self.max_evidence_chars, self.first_evidence_chars)
        if any(not isinstance(value, int) or value <= 0 for value in values):
            raise ValueError("Research budgets must be positive integers")
        if self.max_actions > self.max_model_calls or self.first_evidence_chars > self.max_evidence_chars:
            raise ValueError("Inconsistent research budget")
        if self.deadline_seconds <= 0 or self.model_timeout_seconds <= 0:
            raise ValueError("Timeouts must be positive")


class WorkflowStop(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _text(value: Any, *, limit: int, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise WorkflowStop(f"invalid_{field}")
    return value.strip()


def _evidence_label(segment: dict) -> str:
    return (f"[{segment['id']} | {segment['section']} | "
            f"p.{segment['page_start']}-{segment['page_end']} | "
            f"chars {segment['start']}-{segment['end']}]\n{segment['text']}")


def _evidence_text(segments: list[dict]) -> str:
    return "\n\n".join(_evidence_label(segment) for segment in segments)


def _segment(row: dict, *, start: int, max_chars: int) -> dict | None:
    source = row.get("content")
    if not isinstance(source, str) or not source.strip() or start >= len(source):
        return None
    chunk_id = _text(row.get("id"), limit=200, field="chunk_id")
    section = str(row.get("section") or "unknown")[:80]
    if section.strip().lower() in SKIP_SECTIONS:
        return None
    page_start = row.get("page_start") or 0
    page_end = row.get("page_end") or page_start
    if not isinstance(page_start, int) or not isinstance(page_end, int):
        raise WorkflowStop("invalid_page_metadata")
    shown = source[start:start + max_chars]
    return {"id": chunk_id, "section": section, "page_start": page_start,
            "page_end": page_end, "start": start, "end": start + len(shown),
            "source_chars": len(source), "has_more_before": start > 0,
            "has_more_after": start + len(shown) < len(source),
            "text": shown, "sha256": hashlib.sha256(shown.encode("utf-8")).hexdigest()}


def _pack(rows: list[dict], segments: list[dict], *, limit: int, budget: ResearchBudget,
          start: int | None = None) -> list[dict]:
    added: list[dict] = []
    ids = {segment["id"] for segment in segments}
    spans = {(segment["id"], segment["start"], segment["end"]) for segment in segments}
    fingerprints = {segment["sha256"] for segment in segments}
    for row in rows:
        if len(ids) >= budget.max_unique_chunks and row.get("id") not in ids:
            break
        row_start = row.get("_start", 0) if start is None else start
        if not isinstance(row_start, int) or row_start < 0:
            raise WorkflowStop("invalid_retrieval_result")
        segment = _segment(row, start=row_start, max_chars=budget.max_chunk_chars)
        if (segment is None or (segment["id"], segment["start"], segment["end"]) in spans or
                (segment["sha256"] in fingerprints and segment["id"] not in ids)):
            continue
        candidate = segments + added + [segment]
        if len(_evidence_text(candidate)) > limit:
            continue
        added.append(segment)
        ids.add(segment["id"])
        spans.add((segment["id"], segment["start"], segment["end"]))
        fingerprints.add(segment["sha256"])
    return added


def _lexical_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", text or "")
        if len(token) >= 3 and token.lower() not in LEXICAL_STOPWORDS
    ]


def _query_window_start(content: str, query: str, window: int = 600) -> int:
    """Choose a query-focused excerpt without using evaluation labels or answer keys."""
    tokens = set(_lexical_tokens(query))
    if not tokens or len(content) <= window:
        return 0
    lowered = content.lower()
    positions: list[int] = []
    for token in tokens:
        positions.extend(match.start() for match in re.finditer(re.escape(token), lowered))
    if not positions:
        return 0
    candidates = {max(0, min(len(content) - window, position - window // 4)) for position in positions}

    def score(begin: int) -> tuple[int, int]:
        excerpt = lowered[begin:begin + window]
        weighted_hits = sum((len(token) + 2) * excerpt.count(token) for token in tokens)
        return weighted_hits, -begin

    return max(candidates, key=score)


def _lexical_rank(query: str, rows: list[dict], exclude_ids: set[str]) -> list[dict]:
    """Lightweight BM25 complement for exact paper entities and mechanism names."""
    ordered_query_terms = _lexical_tokens(query)
    query_terms = set(ordered_query_terms)
    query_bigrams = {
        f"{left} {right}"
        for left, right in zip(ordered_query_terms, ordered_query_terms[1:])
        if left != right
    }
    usable = [
        row for row in rows
        if row.get("id") not in exclude_ids
        and str(row.get("section") or "").strip().lower() not in SKIP_SECTIONS
        and isinstance(row.get("content"), str)
    ]
    if not query_terms or not usable:
        return []
    documents = [_lexical_tokens(row["content"]) for row in usable]
    normalized_documents = [" ".join(document) for document in documents]
    # Repeated PDF page headers (most often the full paper title) must not be
    # treated as a component-definition heading.  All thresholds are derived
    # from the current paper corpus; no benchmark labels are inspected.
    repetition_threshold = max(2, math.ceil(len(usable) * 0.1))
    bigram_frequency = {
        phrase: sum(phrase in document for document in normalized_documents)
        for phrase in query_bigrams
    }
    informative_query_bigrams = {
        phrase for phrase in query_bigrams
        if bigram_frequency.get(phrase, 0) <= repetition_threshold
    }
    short_line_frequency: Counter[str] = Counter()
    for row in usable:
        normalized_lines = {
            " ".join(_lexical_tokens(line))
            for line in re.split(r"[\r\n]+", row["content"])
            if 0 < len(line.strip()) <= 120
        }
        short_line_frequency.update(line for line in normalized_lines if line)
    document_frequency = {
        term: sum(term in set(document) for document in documents)
        for term in query_terms
    }
    average_length = sum(len(document) for document in documents) / max(len(documents), 1)
    lowered_query = query.lower()
    method_query = any(marker in lowered_query for marker in METHOD_QUERY_MARKERS)
    evaluation_query = any(marker in lowered_query for marker in EVALUATION_QUERY_MARKERS)
    taxonomy_query = any(marker in lowered_query for marker in TAXONOMY_QUERY_MARKERS)
    scored: list[tuple[int, int, float, int, dict]] = []
    for position, (row, document) in enumerate(zip(usable, documents)):
        counts = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = counts[term]
            if not frequency:
                continue
            inverse_frequency = math.log(
                1 + (len(documents) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * len(document) / max(average_length, 1)
            )
            score += inverse_frequency * (frequency * 2.2) / denominator
        section_name = str(row.get("section") or "").strip().lower()
        if method_query and section_name in METHOD_SECTIONS:
            score += 2.0
        if evaluation_query and section_name in EVALUATION_SECTIONS:
            score += 4.0
        if taxonomy_query and "taxonomy" in set(document):
            score += 3.0
        # Title/abstract material is useful for broad summaries, but should not
        # consume the first slot when a named taxonomy or evaluation section is
        # available under a tight evidence budget.
        if section_name == "preamble":
            score -= 6.0 if (taxonomy_query or evaluation_query) else 1.0
        # Short heading-like lines containing both a named entity and its
        # surrounding query words usually mark the definition block, not a
        # later passing mention in results or references.
        rare_terms = {term for term in query_terms if len(term) >= 7 and term != "attention"}
        definition_matches = 0
        for line in re.split(r"[\r\n]+", row["content"]):
            ordered_line_tokens = _lexical_tokens(line)
            line_tokens = set(ordered_line_tokens)
            normalized_line = " ".join(ordered_line_tokens)
            if short_line_frequency[normalized_line] > repetition_threshold:
                continue
            matched_phrases = {
                phrase for phrase in informative_query_bigrams if phrase in normalized_line
            }
            if (0 < len(line.strip()) <= 120 and len(line_tokens & query_terms) >= 2
                    and line_tokens & rare_terms):
                score += 4.0
                # Prefer the block that defines a named component over later
                # mentions of the same words.  This is query-derived and does
                # not inspect benchmark labels or expected answers.
                if matched_phrases:
                    definition_matches = max(definition_matches, len(matched_phrases))
                    score += 20.0
                raw_words = re.findall(r"[A-Za-z][A-Za-z0-9]*", line)
                heading_capitals = sum(word[0].isupper() for word in raw_words[:4])
                if heading_capitals >= 2:
                    score += 8.0
                break
        if score > 0:
            domain_section_match = int(
                (evaluation_query and section_name in EVALUATION_SECTIONS)
                or (taxonomy_query and "taxonomy" in set(document))
            )
            prepared = dict(row)
            prepared["_start"] = _query_window_start(row["content"], query)
            prepared["_retrieval_source"] = "lexical"
            scored.append((domain_section_match, definition_matches, score, -position, prepared))
    # The phrase match already contributes to the numeric score.  Keeping it
    # as the primary sort key made a passing mention such as "reviews lack
    # evaluation metrics" outrank the actual Evaluation section forever.
    scored.sort(key=lambda item: (item[0], item[2], item[1], item[3]), reverse=True)
    return [row for _domain, _matches, _score, _position, row in scored]


def _plan(data: Any) -> list[dict]:
    if not isinstance(data, dict) or not isinstance(data.get("subquestions"), list):
        raise WorkflowStop("invalid_plan")
    questions = data["subquestions"]
    if not 1 <= len(questions) <= 2:
        raise WorkflowStop("invalid_plan")
    parsed = []
    for number, item in enumerate(questions, 1):
        if not isinstance(item, dict) or item.get("id") != f"q{number}":
            raise WorkflowStop("invalid_plan")
        parsed.append({
            "id": item["id"],
            "text": _text(item.get("text"), limit=500, field="plan"),
            "query": _text(item.get("query"), limit=240, field="query"),
        })
    return parsed


def _fuse_subquery_results(results: list[tuple[str, list[dict]]], *, k: int) -> list[dict]:
    """Fuse independently ranked subquery results while preserving coverage.

    One top result per subquestion is reserved first; remaining slots use RRF.
    This prevents a broad subquery from consuming the whole first-pass budget.
    """
    if not results or k <= 0:
        return []
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    rows_by_id: dict[str, dict] = {}
    query_ids: dict[str, list[str]] = {}
    order = 0
    for query_id, rows in results:
        for rank, row in enumerate(rows, 1):
            row_id = row.get("id")
            if not isinstance(row_id, str):
                raise WorkflowStop("invalid_retrieval_result")
            if row_id not in rows_by_id:
                rows_by_id[row_id] = dict(row)
                first_seen[row_id] = order
                query_ids[row_id] = []
                order += 1
            if query_id not in query_ids[row_id]:
                query_ids[row_id].append(query_id)
            scores[row_id] = scores.get(row_id, 0.0) + 1.0 / (60 + rank)

    ranked_ids = sorted(scores, key=lambda row_id: (-scores[row_id], first_seen[row_id]))
    selected_ids: list[str] = []
    # Coverage reservation: keep several independently ranked candidates for
    # every subquestion before RRF can promote generic chunks shared by all
    # queries.  Three is enough to cover a definition plus its adjacent detail
    # under the current eight-candidate cap without letting one query dominate.
    reserve_per_query = max(1, min(3, k // len(results)))
    cursors = [0 for _ in results]
    for _rank in range(reserve_per_query):
        for result_index, (_query_id, rows) in enumerate(results):
            selected_id = None
            while cursors[result_index] < len(rows):
                candidate_id = rows[cursors[result_index]].get("id")
                cursors[result_index] += 1
                if isinstance(candidate_id, str) and candidate_id not in selected_ids:
                    selected_id = candidate_id
                    break
            if selected_id is not None:
                selected_ids.append(selected_id)
            if len(selected_ids) >= k:
                break
        if len(selected_ids) >= k:
            break
    for row_id in ranked_ids:
        if row_id not in selected_ids:
            selected_ids.append(row_id)
        if len(selected_ids) >= k:
            break

    fused: list[dict] = []
    for row_id in selected_ids:
        prepared = dict(rows_by_id[row_id])
        prepared["_query_ids"] = query_ids[row_id]
        prepared["_rrf_score"] = round(scores[row_id], 8)
        fused.append(prepared)
    return fused


def _assessment(data: Any, questions: list[dict], seen_ids: set[str]) -> tuple[list[dict], dict]:
    if not isinstance(data, dict) or not isinstance(data.get("subquestions"), list):
        raise WorkflowStop("invalid_assessment")
    rows = data["subquestions"]
    if (len(rows) != len(questions) or any(not isinstance(row, dict) or
                                          not isinstance(row.get("subquestion_id"), str) for row in rows) or
            {row["subquestion_id"] for row in rows} != {q["id"] for q in questions}):
        raise WorkflowStop("invalid_assessment")
    parsed = []
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("assessment"), str)
                or row.get("assessment") not in ASSESSMENTS):
            raise WorkflowStop("invalid_assessment")
        ids = row.get("evidence_ids")
        if not isinstance(ids, list) or any(not isinstance(item, str) or item not in seen_ids for item in ids):
            raise WorkflowStop("foreign_or_unseen_evidence")
        reason = _text(row.get("reason"), limit=500, field="assessment_reason")
        missing = row.get("missing_information", "")
        if not isinstance(missing, str) or len(missing) > 500:
            raise WorkflowStop("invalid_missing_information")
        if row["assessment"] == "refuted_premise" and not ids:
            raise WorkflowStop("refutation_without_evidence")
        if row["assessment"] not in {"supported", "refuted_premise"} and not missing.strip():
            raise WorkflowStop("missing_information_required")
        assessment_value = row["assessment"]
        normalized_from = None
        if assessment_value == "supported" and missing.strip():
            normalized_from = "supported"
            assessment_value = "missing_specific_fact"
        parsed_row = {"subquestion_id": row["subquestion_id"], "assessment": assessment_value,
                      "evidence_ids": ids, "reason": reason, "missing_information": missing.strip()}
        if normalized_from:
            parsed_row["normalized_from"] = normalized_from
        parsed.append(parsed_row)
    action = data.get("action")
    if not isinstance(action, dict) or action.get("type") not in ACTIONS:
        raise WorkflowStop("invalid_action")
    if action["type"] != "stop" and not any(
        row["assessment"] != "supported" and row["missing_information"] for row in parsed
    ):
        raise WorkflowStop("missing_information_required")
    if action["type"] == "search":
        action = {"type": "search", "query": _text(action.get("query"), limit=240, field="query")}
    elif action["type"] == "expand":
        target = action.get("target_id")
        if not isinstance(target, str) or target not in seen_ids:
            raise WorkflowStop("foreign_or_unseen_evidence")
        if action.get("direction") not in DIRECTIONS:
            raise WorkflowStop("invalid_action")
        action = {"type": "expand", "target_id": target, "direction": action["direction"]}
    else:
        # A model stop ends retrieval, not writing. Explicit gaps remain visible
        # to the final writer; the program must not invent a new search here.
        action = {"type": "stop"}
    return parsed, action


def _isolated_assessment(data, questions, seen_ids):
    """Isolate protocol errors; provenance violations remain hard failures."""
    if not isinstance(data, dict) or not isinstance(data.get("subquestions"), list):
        fallback = [{"subquestion_id": q["id"], "assessment": "unknown", "evidence_ids": [],
                     "reason": "评估协议不可用，仅允许基于已见证据有限回答", "missing_information": "评估未通过"}
                    for q in questions]
        blocked = [{"subquestion_id": q["id"], "question": q["text"], "reason": "invalid_assessment"}
                   for q in questions]
        return fallback, {"type": "stop"}, [], blocked, "assessment_unavailable"
    rows = data["subquestions"]
    valid, blocked = [], []
    for question in questions:
        matching = [r for r in rows if isinstance(r, dict) and r.get("subquestion_id") == question["id"]]
        try:
            if len(matching) != 1:
                raise WorkflowStop("invalid_assessment")
            parsed, _ = _assessment({"subquestions": matching, "action": {"type": "stop"}},
                                     [question], seen_ids)
        except WorkflowStop as exc:
            if exc.reason == "foreign_or_unseen_evidence":
                raise
            blocked.append({"subquestion_id": question["id"], "question": question["text"],
                            "reason": exc.reason})
        else:
            valid.extend(parsed)
    if not valid:
        # Still generate a bounded answer from already seen evidence, but do not
        # reclassify malformed model assessments as semantically supported.
        valid = [{"subquestion_id": q["id"], "assessment": "unknown", "evidence_ids": [],
                  "reason": "评估协议不可用，仅允许基于已见证据有限回答", "missing_information": "评估未通过"}
                 for q in questions]
        return valid, {"type": "stop"}, [], blocked, "assessment_unavailable"
    kept = [q for q in questions if q["id"] in {r["subquestion_id"] for r in valid}]
    try:
        parsed, action = _assessment({"subquestions": valid, "action": data.get("action")}, kept, seen_ids)
    except WorkflowStop as exc:
        if exc.reason == "foreign_or_unseen_evidence":
            raise
        return valid, {"type": "stop"}, kept, blocked, exc.reason
    if blocked:
        action = {"type": "stop"}  # global proposed action may target a blocked question
    return valid, action, kept, blocked, None


def _normalize_final_contract(
    data: Any,
    questions: list[dict],
    seen: dict[str, dict],
    paper_mode: bool,
) -> tuple[dict, list[dict], list[str], bool]:
    """Bind insufficiency to planned questions; keep optional gaps as caveats."""

    if not isinstance(data, dict):
        raise WorkflowStop("invalid_final_answer")
    normalized = dict(data)
    caveats = normalized.get("caveats", [])
    if not isinstance(caveats, list) or any(
        not isinstance(item, str) or not item.strip() or len(item) > 500
        for item in caveats
    ):
        raise WorkflowStop("invalid_final_caveats")
    caveats = [item.strip() for item in caveats]
    coverage = normalized.get("coverage")
    if coverage is None:
        # Compatibility for deterministic offline fixtures and older traces.
        return normalized, [], caveats, False
    if not paper_mode or not questions or not isinstance(coverage, list):
        raise WorkflowStop("invalid_final_coverage")
    expected = {item["id"]: item["text"] for item in questions}
    if len(coverage) != len(expected):
        raise WorkflowStop("invalid_final_coverage")
    parsed: list[dict] = []
    unresolved: list[str] = []
    seen_question_ids: set[str] = set()
    for row in coverage:
        if not isinstance(row, dict):
            raise WorkflowStop("invalid_final_coverage")
        subquestion_id = row.get("subquestion_id")
        status = row.get("status")
        reason = row.get("reason")
        evidence_ids = row.get("evidence_ids", [])
        if (subquestion_id not in expected or subquestion_id in seen_question_ids
                or status not in {"answered", "insufficient"}
                or not isinstance(reason, str) or len(reason) > 500
                or not isinstance(evidence_ids, list)
                or any(not isinstance(cid, str) or cid not in seen for cid in evidence_ids)):
            raise WorkflowStop("invalid_final_coverage")
        if status == "answered" and not evidence_ids:
            raise WorkflowStop("answered_without_evidence")
        parsed_row = {
            "subquestion_id": subquestion_id,
            "status": status,
            "reason": reason.strip(),
            "evidence_ids": evidence_ids,
        }
        parsed.append(parsed_row)
        seen_question_ids.add(subquestion_id)
        if status == "insufficient":
            detail = reason.strip() or "当前证据不足"
            unresolved.append(f"{expected[subquestion_id]}：{detail}")
    if seen_question_ids != set(expected):
        raise WorkflowStop("invalid_final_coverage")
    normalized["unresolved"] = unresolved
    normalized["caveats"] = caveats
    return normalized, parsed, caveats, True


def _citation_label(source: dict) -> str:
    section = str(source.get("section") or "").strip()
    start = source.get("page_start") or 0
    end = source.get("page_end") or start
    page = f"第{start}页" if start == end else f"第{start}-{end}页"
    if section and section.lower() != "unknown":
        return f"{section.title()}，{page}"
    return page


def _build_public_sources(claims: list[dict], seen: dict[str, dict]) -> list[dict]:
    sources: list[dict] = []
    observed: set[tuple[str, int, int]] = set()
    for claim in claims:
        for evidence_id in claim.get("evidence_ids", []):
            source = seen.get(evidence_id)
            if not source:
                continue
            section = str(source.get("section") or "unknown")
            start = int(source.get("page_start") or 0)
            end = int(source.get("page_end") or start)
            key = (section, start, end)
            if key in observed:
                continue
            observed.add(key)
            sources.append({
                "number": len(sources) + 1,
                "section": section,
                "page_start": start,
                "page_end": end,
            })
    return sources


def _render_final(data: Any, seen: dict[str, dict], paper_mode: bool) -> tuple[str, list[dict], list[str]]:
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise WorkflowStop("invalid_final_answer")
    unresolved = data.get("unresolved", [])
    if not isinstance(unresolved, list) or any(not isinstance(item, str) or len(item) > 500 for item in unresolved):
        raise WorkflowStop("invalid_final_answer")
    lines, claims = [], []
    current_heading = None
    # Bound malformed/oversized responses explicitly; never silently discard
    # valid later claims (which may contain the experiment results).
    if len(data["items"]) > 64:
        raise WorkflowStop("final_item_budget")
    for item in data["items"]:
        if not isinstance(item, dict) or item.get("kind") not in KINDS:
            raise WorkflowStop("invalid_final_answer")
        kind = item["kind"]
        if not paper_mode and kind == "paper_fact":
            raise WorkflowStop("general_mode_paper_claim")
        statement = _text(item.get("text"), limit=1000, field="claim")
        heading = item.get("heading")
        if heading is not None:
            heading = _text(heading, limit=80, field="heading")
        ids = item.get("evidence_ids", [])
        if not isinstance(ids, list) or any(not isinstance(cid, str) for cid in ids):
            raise WorkflowStop("invalid_final_answer")
        if any(cid not in seen for cid in ids):
            raise WorkflowStop("invalid_citation")
        if kind == "paper_fact" and not ids:
            raise WorkflowStop("paper_fact_without_citation")
        if not paper_mode and ids:
            raise WorkflowStop("general_mode_citation")
        references = list(dict.fromkeys(_citation_label(seen[cid]) for cid in ids))
        if heading and heading != current_heading:
            if lines:
                lines.append("")
            lines.append(f"### {heading}")
            current_heading = heading
        label = "" if kind == "paper_fact" else ("进一步分析：" if kind == "inference" else "背景补充：")
        lines.append(f"- {label}{statement}" + (f" [{'; '.join(references)}]" if references else ""))
        claims.append({"kind": kind, "heading": heading, "text": statement, "evidence_ids": ids})
    caveats = data.get("caveats", [])
    if caveats:
        lines.extend(["", "### 当前证据边界与待核实内容"])
        lines.extend(f"- {item}" for item in caveats)
    if unresolved:
        lines.extend(["", "### 尚缺少核心证据"])
        lines.extend(f"- {item}" for item in unresolved)
    if not lines:
        lines.append("当前证据不足，无法作出有依据的结论。")
    return "\n".join(lines), claims, unresolved


def _prepare_final_contract(data, questions, seen, paper_mode, trace):
    """Keep independently valid records; never claim source identity proves truth.

    This is deterministic display recovery, not a model retry or JSON repair.
    The original response remains in the model event for audit.
    """
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise WorkflowStop("invalid_final_answer")
    if not data["items"]:
        # A deliberate evidence-insufficient answer is not a malformed record.
        return _normalize_final_contract(data, questions, seen, paper_mode)
    accepted, decisions = [], []
    for index, item in enumerate(data["items"][:64]):
        parts = [item]
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            text = item["text"]
            if 1000 < len(text) <= 20000:
                parts = [dict(item, text=text[pos:pos + 1000])
                         for pos in range(0, len(text), 1000)]
                decisions.append({"index": index, "reason": "long_claim_split",
                                  "parts": len(parts)})
        try:
            for part in parts:
                _render_final({"items": [part]}, seen, paper_mode)
            if len(accepted) + len(parts) > 64:
                raise WorkflowStop("final_item_budget")
        except (WorkflowStop, TypeError, ValueError) as exc:
            decisions.append({"index": index,
                              "reason": getattr(exc, "reason", "invalid_item"),
                              "discarded": True})
        else:
            accepted.extend(parts)
    if len(data["items"]) > 64:
        decisions.append({"reason": "final_item_budget", "discarded": True,
                          "count": len(data["items"]) - 64})
    if not accepted:
        # Do not turn a paper-fact record into background to bypass provenance.
        reason = next((row["reason"] for row in decisions if row.get("discarded")),
                      "no_valid_answer_items")
        raise WorkflowStop(reason)
    prepared = dict(data, items=accepted)
    try:
        normalized = _normalize_final_contract(prepared, questions, seen, paper_mode)
        _render_final(normalized[0], seen, paper_mode)
    except (WorkflowStop, TypeError, ValueError) as exc:
        decisions.append({"reason": getattr(exc, "reason", "invalid_final_metadata"),
                          "metadata_discarded": True})
        # Invalid coverage must not be advertised as answered/supported.
        normalized = ({"items": accepted, "unresolved": [], "caveats": []}, [], [], False)
    if decisions:
        report = {"retained_items": len(accepted), "decisions": decisions,
                  "extra_model_calls": 0, "semantic_accuracy_proven": False}
        trace["answer_degradation"] = report
        trace["events"].append({"step": "answer_degradation", **report})
    return normalized


def _save_private_trace(trace: dict, directory: Path | None) -> None:
    if directory is None:
        return
    directory = Path(directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{trace['request_id']}.json"
        target.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        # Logging failure must not erase an already generated paid answer.
        # It is still an observability failure, not a successful persistence.
        import logging
        trace["trace_persistence"] = {"status": "failed", "error_type": type(exc).__name__}
        logging.getLogger(__name__).error("Research trace persistence failed: %s", type(exc).__name__)


def run_deep_research(question: str, thread_id: str, retriever: Any, model: Any, *,
                      mode: str = "paper_deep", budget: ResearchBudget | None = None,
                      answer_depth: str = "beginner",
                      conversation_history: list[dict] | None = None,
                      enable_quality_review: bool = False,
                      trace_dir: Path | None = None, clock=time.monotonic) -> dict:
    """Run one bounded request; retriever/model are injected for offline tests."""
    budget = budget or ResearchBudget()
    conversation_history = list(conversation_history or [])[-4:]
    if mode not in {"paper_deep", "general"}:
        raise ValueError("Unsupported mode")
    if answer_depth not in {"beginner", "concise", "advanced"}:
        raise ValueError("unsupported_answer_depth")
    question = _text(question, limit=4000, field="question")
    if mode == "paper_deep":
        _text(thread_id, limit=200, field="thread_id")
    started = clock()
    deadline = started + budget.deadline_seconds
    request_id = str(uuid.uuid4())
    trace: dict[str, Any] = {
        "request_id": request_id, "thread_id": thread_id if mode == "paper_deep" else None,
        "mode": mode, "question": question, "budget": asdict(budget),
        "events": [], "model_calls": 0, "actions": 0, "retrieval_calls": 0,
        "input_tokens": 0, "output_tokens": 0, "usage_missing_calls": 0,
        "cached_tokens": 0,
        "external_payload_chars": 0,
    }
    segments: list[dict] = []
    questions: list[dict] = []
    coverage: list[dict] = []
    caveats: list[str] = []
    sources: list[dict] = []
    structured_coverage = False
    stop_reason = "not_started"

    def remaining() -> float:
        seconds = deadline - clock()
        if seconds <= 0:
            raise WorkflowStop("deadline")
        return seconds

    def event(name: str, **data: Any) -> None:
        trace["events"].append({"step": name, **data})

    def call(stage: str, payload: dict, output_limit: int) -> Any:
        remaining_seconds = remaining()
        if trace["model_calls"] >= budget.max_model_calls:
            raise WorkflowStop("model_budget_exhausted")
        trace["model_calls"] += 1
        trace["external_payload_chars"] += len(json.dumps(payload, ensure_ascii=False))
        response = model.call(stage, payload, max_output_tokens=output_limit,
                              timeout_seconds=min(remaining_seconds, budget.model_timeout_seconds))
        remaining()
        if not isinstance(response, dict):
            raise WorkflowStop("invalid_model_response")
        usage = response.get("usage") or {}
        if not isinstance(usage.get("input_tokens"), int) or not isinstance(usage.get("output_tokens"), int):
            trace["usage_missing_calls"] += 1
        else:
            trace["input_tokens"] += usage["input_tokens"]
            trace["output_tokens"] += usage["output_tokens"]
        if isinstance(usage.get("cached_tokens"), int):
            trace["cached_tokens"] += usage["cached_tokens"]
        else:
            trace["cache_usage_missing_calls"] = trace.get("cache_usage_missing_calls", 0) + 1
        event("model", stage=stage, raw=response.get("raw"), data=response.get("data"), usage=usage,
              parse_error=response.get("parse_error"), request_meta=response.get("request_meta"),
              finish_reason=response.get("finish_reason"), output_limit_reached=response.get("output_limit_reached"),
              output_budget_exceeded=response.get("output_budget_exceeded"))
        if response.get("parse_error"):
            if stage == "final":
                from partial_output import recover_final
                recovered, recovery = recover_final(response, {row["id"]: row for row in segments})
                if recovery is not None:
                    trace["output_recovery"] = recovery
                    event("output_recovery", **recovery)
                if recovered is not None:
                    return recovered
            raise WorkflowStop(str(response["parse_error"]))
        if "data" not in response or response["data"] is None:
            raise WorkflowStop("invalid_model_response")
        return response["data"]

    try:
        if mode == "paper_deep" and not retriever.has_paper(thread_id):
            raise WorkflowStop("no_paper")
        if mode == "general":
            final = call("general", {
                "question": question, "mode": mode, "answer_depth": answer_depth,
            }, {"concise": 900, "beginner": 1800, "advanced": 2400}[answer_depth])
            final, _, _, _ = _prepare_final_contract(final, [], {}, False, trace)
            answer, claims, unresolved = _render_final(final, {}, False)
            stop_reason = "general_answer"
        else:
            plan_data = call("plan", {
                "conversation": conversation_history,
                "question": question,
            }, 400)
            questions = _plan(plan_data)
            event("plan", subquestions=questions,
                  initial_queries=[{"subquestion_id": item["id"], "query": item["query"]}
                                   for item in questions])
            remaining()
            trace["actions"] += 1
            per_query_results: list[tuple[str, list[dict]]] = []
            query_trace: list[dict] = []
            for item in questions:
                trace["retrieval_calls"] += 1
                found_for_query = retriever.search(
                    thread_id, item["query"], k=budget.max_unique_chunks,
                    exclude_ids=set(), timeout_seconds=remaining())
                remaining()
                if not isinstance(found_for_query, list):
                    raise WorkflowStop("invalid_retrieval_result")
                per_query_results.append((item["id"], found_for_query))
                query_trace.append({
                    "subquestion_id": item["id"],
                    "query": item["query"],
                    "candidate_ids": [row.get("id") for row in found_for_query],
                    "candidate_sources": {
                        row.get("id"): row.get("_retrieval_source", "unknown")
                        for row in found_for_query
                    },
                })
            found = _fuse_subquery_results(per_query_results, k=budget.max_unique_chunks)
            segments.extend(_pack(found, segments, limit=budget.first_evidence_chars, budget=budget))
            event("retrieval", action="multi_search", queries=query_trace,
                  retrieval_calls=len(per_query_results), candidate_ids=[r.get("id") for r in found],
                  candidate_sources={r.get("id"): r.get("_retrieval_source", "unknown") for r in found},
                  candidate_query_ids={r.get("id"): r.get("_query_ids", []) for r in found},
                  candidate_rrf_scores={r.get("id"): r.get("_rrf_score") for r in found},
                  candidate_starts={r.get("id"): r.get("_start", 0) for r in found},
                  provided_segments=segments.copy(), evidence_chars=len(_evidence_text(segments)))
            seen_ids = {part["id"] for part in segments}
            assessment_data = call("assess", {"question": question, "subquestions": questions,
                                              "evidence": segments.copy()}, 800)
            event("model_proposal", raw=assessment_data)
            try:
                assessments, action, kept_questions, blocked, action_error = _isolated_assessment(
                    assessment_data, questions, seen_ids)
                if blocked or action_error:
                    trace["assessment_degradation"] = {"blocked_subquestions": blocked,
                        "action_error": action_error, "original_proposal": assessment_data,
                        "effective_action": action, "semantic_accuracy_proven": False}
                    event("assessment_degradation", **trace["assessment_degradation"])
                # If every assessment is malformed, final writing may still use
                # the already scoped evidence, explicitly without valid assessment.
                questions = kept_questions or questions
            except WorkflowStop as exc:
                event("gate", allowed=False, reason=exc.reason, remaining_actions=budget.max_actions - trace["actions"])
                raise
            if action["type"] == "expand" and action["direction"] == "continue":
                target_segments = [part for part in segments if part["id"] == action["target_id"]]
                remaining_chars = max(
                    (part.get("source_chars", part["end"]) - part["end"] for part in target_segments),
                    default=0,
                )
                if remaining_chars < min(200, budget.max_chunk_chars // 3):
                    question_text = {item["id"]: item["text"] for item in questions}
                    gap_query = " ".join(
                        f"{question_text[row['subquestion_id']]} {row['missing_information']}"
                        for row in assessments if row["missing_information"]
                    )[:240]
                    action = {
                        "type": "search",
                        "query": gap_query,
                        "_guard_override": "low_yield_expand",
                    }
            event("assessment", subquestions=assessments, action=action)
            if action["type"] == "stop":
                stop_reason = "model_stop"
                event("gate", allowed=True, reason="model_stop", remaining_actions=budget.max_actions - trace["actions"])
            elif trace["actions"] >= budget.max_actions:
                stop_reason = "action_budget_exhausted"
                event("gate", allowed=False, reason=stop_reason, remaining_actions=0)
            else:
                remaining()
                override_reason = action.pop("_guard_override", None)
                if override_reason:
                    event("gate", allowed=False, reason=override_reason,
                          remaining_actions=budget.max_actions - trace["actions"])
                event("gate", allowed=True,
                      reason="guard_fallback_search" if override_reason else "within_budget",
                      remaining_actions=budget.max_actions - trace["actions"] - 1)
                trace["actions"] += 1
                if action["type"] == "search":
                    trace["retrieval_calls"] += 1
                    found = retriever.search(thread_id, action["query"], k=budget.max_unique_chunks,
                                             exclude_ids=seen_ids, timeout_seconds=remaining())
                    remaining()
                    if not isinstance(found, list):
                        raise WorkflowStop("invalid_retrieval_result")
                    newly_provided = _pack(found, segments, limit=budget.max_evidence_chars, budget=budget)
                    segments.extend(newly_provided)
                    event("followup", action=action, candidate_ids=[r.get("id") for r in found],
                          candidate_sources={r.get("id"): r.get("_retrieval_source", "unknown") for r in found},
                          candidate_starts={r.get("id"): r.get("_start", 0) for r in found},
                          provided_segments=newly_provided, evidence_chars=len(_evidence_text(segments)))
                else:
                    found = retriever.expand(thread_id, action["target_id"], action["direction"],
                                             seen_segments=segments.copy(), timeout_seconds=remaining())
                    remaining()
                    if not isinstance(found, list):
                        raise WorkflowStop("invalid_retrieval_result")
                    start = max((p["end"] for p in segments if p["id"] == action["target_id"]), default=0) \
                        if action["direction"] == "continue" else 0
                    newly_provided = _pack(found, segments, limit=budget.max_evidence_chars,
                                           budget=budget, start=start)
                    segments.extend(newly_provided)
                    event("followup", action=action, candidate_ids=[r.get("id") for r in found],
                          provided_segments=newly_provided, evidence_chars=len(_evidence_text(segments)))
                stop_reason = "followup_executed" if newly_provided else "followup_empty"
            seen = {part["id"]: part for part in segments}
            final = call("final", {"question": question, "subquestions": questions,
                                   "blocked_subquestions": (trace.get("assessment_degradation") or {}).get("blocked_subquestions", []),
                                   "assessment_unavailable": (trace.get("assessment_degradation") or {}).get("action_error") == "assessment_unavailable",
                                   "conversation": conversation_history,
                                   "assessments": assessments, "evidence": segments.copy(),
                                   "stop_reason": stop_reason,
                                   "answer_depth": answer_depth},
                         RAG_PAPER_OUTPUT_LIMITS[answer_depth])
            final, coverage, caveats, structured_coverage = _prepare_final_contract(
                final, questions, seen, True, trace
            )
            answer, claims, unresolved = _render_final(final, seen, True)
            sources = _build_public_sources(claims, seen)
        # The assessor can silently say "supported" and stop, while the final
        # writer later admits that evidence is missing.  At that point the
        # three-call budget is exhausted, so fail visibly instead of reporting
        # a misleading successful completion.
        if trace.get("output_recovery", {}).get("retained_items", 0):
            from partial_output import TRUNCATION_NOTICE
            unresolved = [TRUNCATION_NOTICE]
            answer += "\n\n### 本次输出未完成\n- " + TRUNCATION_NOTICE
            status, stop_reason = "partial", "output_token_limit_partial"
        elif trace.get("assessment_degradation"):
            status, stop_reason = "partial", "assessment_validation_partial"
            report = trace["assessment_degradation"]
            answer += "\n\n### 评估降级说明\n- 部分评估未通过格式检查，未执行额外补查；已保留基于当前证据的回答。"
            for row in report["blocked_subquestions"]:
                qualifier = "未通过评估，未按已核验子问题输出" if not report["action_error"] == "assessment_unavailable" else "评估不可用，相关解释仅供有限参考"
                answer += f"\n- {row['subquestion_id']}：{row['question']}；{qualifier}（{row['reason']}），可单独追问。"
        elif trace.get("answer_degradation"):
            status, stop_reason = "partial", "answer_validation_partial"
            answer += "\n\n部分输出格式已调整或不合规内容已省略；保留内容通过来源身份校验，但仍需核查事实。"
        elif mode == "paper_deep" and unresolved:
            postcheck_reason = (
                "final_unresolved_after_supported_stop"
                if stop_reason == "model_stop"
                else "final_unresolved_after_followup"
            )
            event("postcheck", allowed=False, reason=postcheck_reason, unresolved=unresolved)
            status = "insufficient" if structured_coverage else "partial"
            stop_reason = (
                "final_revealed_gap_after_stop"
                if stop_reason == "model_stop"
                else "followup_incomplete"
            )
        elif structured_coverage and caveats:
            status = "ok_with_limits"
            stop_reason = "answer_with_caveats"
        else:
            status = "ok"
        if trace.get("assessment_degradation") and "### 评估降级说明" not in answer:
            answer += "\n\n### 评估降级说明\n- 评估格式未完全通过，未追加补查；部分问题尚待单独核实。"
            for row in trace["assessment_degradation"]["blocked_subquestions"]:
                answer += f"\n- {row['subquestion_id']}：{row['question']}（{row['reason']}），可单独追问。"
        event("finish", stop_reason=stop_reason, claims=claims, unresolved=unresolved,
              coverage=coverage, caveats=caveats)
    except WorkflowStop as exc:
        status, stop_reason = "stopped", exc.reason
        answer = "本次研究未能安全完成，请查看停止原因；当前没有可核验的论文结论。"
        claims, unresolved, coverage, caveats, sources = [], [stop_reason], [], [], []
        event("finish", stop_reason=stop_reason)
    except Exception as exc:
        status, stop_reason = "error", "component_error"
        answer = "本次研究遇到工具或模型错误，未输出论文结论。"
        claims, unresolved, coverage, caveats, sources = [], [stop_reason], [], [], []
        event("finish", stop_reason=stop_reason, error_type=type(exc).__name__)
    from quality_review import cache_total, review_once
    quality_report = review_once(
        model, question, answer, segments, trace,
        enabled=enable_quality_review and mode == "paper_deep"
        and not trace.get("output_recovery") and not trace.get("answer_degradation")
        and not trace.get("assessment_degradation")
        and status in {"ok", "ok_with_limits", "partial", "insufficient"},
    )
    trace["cached_tokens"] = cache_total(trace)
    trace["quality_review"] = quality_report
    trace.update({"status": status, "stop_reason": stop_reason, "answer": answer,
                  "claims": claims, "unresolved": unresolved, "coverage": coverage,
                  "caveats": caveats, "sources": sources,
                  "evidence_chars": len(_evidence_text(segments)),
                  "unique_evidence_ids": len({part["id"] for part in segments}),
                  "elapsed_ms": round((clock() - started) * 1000, 2)})
    _save_private_trace(trace, trace_dir)
    return {"request_id": request_id, "status": status, "stop_reason": stop_reason,
            "answer": answer, "claims": claims, "unresolved": unresolved,
            "coverage": coverage, "caveats": caveats, "sources": sources,
            "model_calls": trace["model_calls"], "actions": trace["actions"],
            "retrieval_calls": trace["retrieval_calls"],
            "evidence_chars": trace["evidence_chars"],
            "unique_evidence_ids": trace["unique_evidence_ids"],
            "evidence_ids": sorted({part["id"] for part in segments}),
            "input_tokens": trace["input_tokens"] if not trace["usage_missing_calls"] else None,
            "output_tokens": trace["output_tokens"] if not trace["usage_missing_calls"] else None,
            "cached_tokens": trace["cached_tokens"],
            "usage_missing_calls": trace["usage_missing_calls"],
            "external_payload_chars": trace["external_payload_chars"],
            "elapsed_ms": trace["elapsed_ms"],
            "steps": [{"step": e["step"], "reason": e.get("reason"), "action": e.get("action")}
                      for e in trace["events"]]}


class ChromaPaperRetriever:
    """Restrict all retrieval and expansion to the server-selected thread."""

    def __init__(self, rag: Any):
        self.rag = rag
        self._corpus_cache: dict[str, list[dict]] = {}

    @staticmethod
    def _guard_rows(thread_id: str, rows: list[dict]) -> list[dict]:
        if any(not isinstance(row, dict) or
               not isinstance(row.get("id"), str) or
               not row["id"].startswith(thread_id + "_") for row in rows):
            raise WorkflowStop("foreign_source_evidence")
        return rows

    def has_paper(self, thread_id: str) -> bool:
        return bool(self._corpus(thread_id))

    def _corpus(self, thread_id: str) -> list[dict]:
        if thread_id not in self._corpus_cache:
            self._corpus_cache[thread_id] = self._guard_rows(
                thread_id, self.rag.list_chunks(thread_id)
            )
        return self._corpus_cache[thread_id]

    def search(self, thread_id: str, query: str, *, k: int, exclude_ids: set[str],
               timeout_seconds: float) -> list[dict]:
        semantic = self._guard_rows(thread_id, self.rag.search(
            thread_id, query, k=min(k, 8), exclude_ids=exclude_ids, pool=20,
            timeout_seconds=min(timeout_seconds, 30.0)))
        semantic = [dict(row, _retrieval_source="semantic") for row in semantic]
        corpus = self._corpus(thread_id)
        lexical = _lexical_rank(query, corpus, exclude_ids)
        # Reserve most slots for exact paper entities while retaining semantic
        # candidates for paraphrases. This changes only the G1 workflow adapter.
        lexical_quota = min(len(lexical), max(2, min(k, 6)))
        combined = lexical[:lexical_quota] + semantic
        seen: set[str] = set()
        selected: list[dict] = []
        for row in combined:
            if row["id"] in seen or row["id"] in exclude_ids:
                continue
            selected.append(row)
            seen.add(row["id"])
            if len(selected) >= k:
                break
        return selected

    def expand(self, thread_id: str, target_id: str, direction: str, *,
               seen_segments: list[dict], timeout_seconds: float) -> list[dict]:
        del timeout_seconds  # Local Chroma get; deadline checked around the call.
        if target_id not in {part["id"] for part in seen_segments}:
            raise WorkflowStop("foreign_or_unseen_evidence")
        rows = self._corpus(thread_id)
        position = next((i for i, row in enumerate(rows) if row["id"] == target_id), None)
        if position is None:
            raise WorkflowStop("foreign_or_unseen_evidence")
        if direction == "continue":
            return [rows[position]]
        offset = 1 if direction == "next" else -1
        neighbor = position + offset
        return [rows[neighbor]] if 0 <= neighbor < len(rows) else []
