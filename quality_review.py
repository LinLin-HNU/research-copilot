"""One advisory paper review. No scores, tool loops, or answer rewriting."""

import json
import re
import hashlib


def locate_quote(text, quote):
    """Return auditable source offsets; only whitespace may be ignored."""
    start = text.find(quote) if quote else -1
    if start >= 0:
        return {"start": start, "end": start + len(quote), "normalization": "exact",
                "original_quote": text[start:start + len(quote)]}
    chars, offsets = [], []
    for index, char in enumerate(text):
        if not char.isspace():
            chars.append(char)
            offsets.append(index)
    normalized = "".join(char for char in quote if not char.isspace())
    start = "".join(chars).find(normalized) if normalized else -1
    if start < 0:
        return None
    left, right = offsets[start], offsets[start + len(normalized) - 1] + 1
    return {"start": left, "end": right, "normalization": "whitespace_only",
            "original_quote": text[left:right]}


def claim_table(answer, trace):
    finish = next((event for event in reversed(trace.get("events", []))
                   if event.get("step") == "finish"), {})
    claims = [{"claim_id": f"claim_{index}", "text": row["text"], "kind": row.get("kind")}
              for index, row in enumerate(finish.get("claims") or [], 1)]
    claims.extend({"claim_id": f"caveat_{index}", "text": text, "kind": "caveat"}
                  for index, text in enumerate(finish.get("caveats") or [], 1))
    if not claims:
        claims = [{"claim_id": "claim_1", "text": answer, "kind": "draft"}]
    return claims


def validate_findings(issues, answer, selected, claims, *, legacy=False):
    """Validate transport/provenance, never claim semantic correctness."""
    sources = {row["id"]: row["text"] for row in selected}
    targets = {row["claim_id"]: row for row in claims}
    valid, decisions = [], []
    for index, issue in enumerate(issues, 1):
        decision = {"finding_id": f"finding_{index}", "issue": issue,
                    "accepted": False, "reason": "invalid_schema"}
        decisions.append(decision)
        if not isinstance(issue, dict):
            continue
        cid, quote, ids = issue.get("claim_id"), issue.get("source_quote"), issue.get("evidence_ids")
        reason, category = issue.get("reason"), issue.get("category")
        if (not isinstance(category, str) or category not in {"number", "causality", "lineage", "absence", "coverage", "attribution"}
                or (cid is not None and not isinstance(cid, str))
                or not isinstance(reason, str) or not reason.strip() or len(reason) > 2000
                or not isinstance(ids, list) or not all(isinstance(eid, str) for eid in ids)
                or not isinstance(quote, str) or len(quote) > 500):
            continue
        if cid == "document" and category != "coverage":
            decision["reason"] = "claim_not_found"
            continue
        if cid not in targets and cid != "document":
            old_quote = issue.get("draft_quote")
            if not legacy or not isinstance(old_quote, str) or (old_quote and old_quote not in answer):
                decision["reason"] = "claim_not_found"
                continue
            cid = next((row["claim_id"] for row in claims if old_quote and old_quote in row["text"]), "legacy_unspecified")
        if any(eid not in sources for eid in ids):
            decision["reason"] = "evidence_out_of_scope"
            continue
        if not ids or not quote.strip():
            decision["reason"] = "source_not_provided"
            continue
        matched = next(((eid, locate_quote(sources[eid], quote)) for eid in ids
                        if locate_quote(sources[eid], quote) is not None), None)
        if matched is None:
            decision["reason"] = "quote_mismatch"
            continue
        source_id, location = matched
        decision.update(accepted=True, reason="source_text_matched", claim_id=cid,
                        source_id=source_id, source_location=location)
        valid.append({**issue, "claim_id": cid, "source_location": location,
                      "source_id": source_id, "verification": "model_flag_requires_review"})
    return valid, decisions


def add_usage(trace, usage):
    usage = usage if isinstance(usage, dict) else {}
    if all(isinstance(usage.get(key), int) for key in ("input_tokens", "output_tokens")):
        for key in ("input_tokens", "output_tokens"):
            trace[key] += usage[key]
    else:
        trace["usage_missing_calls"] += 1
    cached = usage.get("cached_tokens")
    if isinstance(cached, int):
        trace["cached_tokens"] += cached
    else:
        trace["cache_usage_missing_calls"] = trace.get("cache_usage_missing_calls", 0) + 1


def cache_total(trace):
    return None if trace.get("cache_usage_missing_calls", 0) else trace.get("cached_tokens", 0)


def review_once(model, question, answer, evidence, trace, *, enabled=False):
    """Select bounded excerpts; findings are advisory and retain their provenance."""
    if not enabled:
        return {"status": "disabled", "issues": [], "rewrite_calls": 0}
    if len(answer) > 10000:
        return {"status": "skipped_draft_limit", "issues": [], "rewrite_calls": 0}
    terms = set(re.findall(r"[a-z][a-z0-9_-]{3,}", (question + " " + answer).lower()))
    ranked = sorted(enumerate(evidence), key=lambda pair: (
        -len(terms & set(re.findall(r"[a-z][a-z0-9_-]{3,}", pair[1].get("text", "").lower()))),
        pair[0],
    ))
    selected, remaining = [], 12000
    for _, row in ranked:
        if remaining <= 0 or len(selected) >= 12:
            break
        text = row.get("text", "")[:min(1600, remaining)]
        if text:
            selected.append({"id": row["id"], "section": row.get("section"), "text": text,
                             "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                             "source_start": 0, "source_end": len(text)})
            remaining -= len(text)
    if not selected:
        return {"status": "skipped_no_evidence", "issues": [], "rewrite_calls": 0}
    claims = claim_table(answer, trace)
    payload = {"question": question[:2000], "draft": answer, "draft_claims": claims, "evidence": selected,
               "scope": "Selected excerpts only; absence in excerpts is not absence in the paper."}
    event = {"step": "quality_review", "provided_evidence_ids": [r["id"] for r in selected],
             "evidence_chars": 12000 - remaining, "rewrite_calls": 0,
             "provided_segments": selected, "draft_claims": claims,
             "limits": {"max_calls": 1, "max_output_tokens": 1200, "timeout_seconds": 30,
                        "max_evidence_chars": 12000, "max_draft_chars": 10000}}
    trace["model_calls"] += 1
    trace["external_payload_chars"] += len(json.dumps(payload, ensure_ascii=False))
    trace["events"].append(event)
    try:
        response = model.call("quality_review", payload, max_output_tokens=1200, timeout_seconds=30.0)
    except Exception as exc:
        add_usage(trace, {})
        event.update(status="failed", error_type=type(exc).__name__)
        return {"status": "failed", "issues": [], "rewrite_calls": 0}
    if not isinstance(response, dict):
        add_usage(trace, {})
        event["status"] = "invalid_response"
        return {"status": "invalid_response", "issues": [], "rewrite_calls": 0}
    usage = response.get("usage") or {}
    add_usage(trace, usage)
    event.update(usage=usage, raw=response.get("raw"), parse_error=response.get("parse_error"),
                 finish_reason=response.get("finish_reason"), output_limit_reached=response.get("output_limit_reached"))
    data = response.get("data")
    issues = data.get("issues") if isinstance(data, dict) else None
    if response.get("parse_error") or not isinstance(issues, list) or len(issues) > 6:
        event["status"] = "invalid_response"
        return {"status": "invalid_response", "issues": [], "rewrite_calls": 0}
    valid, decisions = validate_findings(issues, answer, selected, claims)
    rejected = len(issues) - len(valid)
    report = {"status": "reviewed" if not rejected else "reviewed_with_rejected_findings",
              "issues": valid, "rejected_findings": rejected, "rewrite_calls": 0,
              "finding_decisions": decisions,
              "scope": "excerpt_review_not_full_fact_verification"}
    event.update(report)
    return report
