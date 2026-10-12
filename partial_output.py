"""Recover only closed, validated answer records after a length stop.

No suffix completion, sentence repair, model calls or semantic verification.
"""
import json

TRUNCATION_NOTICE = "本次回答达到输出长度上限，仅保留完整且通过来源身份校验的段落；未完成部分请继续追问。"


def complete_items(raw):
    if not isinstance(raw, str) or len(raw) > 200000:
        return []
    text = raw.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline < 0:
            return []
        text = text[first_newline + 1:]
    decoder = json.JSONDecoder()

    def skip(index):
        while index < len(text) and text[index].isspace():
            index += 1
        return index

    pos = skip(0)
    if pos >= len(text) or text[pos] != "{":
        return []
    pos += 1
    keys = set()
    try:
        for _ in range(20):
            pos = skip(pos)
            key, pos = decoder.raw_decode(text, pos)
            if not isinstance(key, str) or key in keys:
                return []
            keys.add(key)
            pos = skip(pos)
            if pos >= len(text) or text[pos] != ":":
                return []
            pos = skip(pos + 1)
            if key == "items":
                if pos >= len(text) or text[pos] != "[":
                    return []
                pos += 1
                found = []
                while True:
                    pos = skip(pos)
                    if pos >= len(text) or text[pos] == "]":
                        return found
                    try:
                        item, end = decoder.raw_decode(text, pos)
                    except (ValueError, RecursionError):
                        return found
                    if len(found) >= 64:
                        return []
                    found.append(item)
                    pos = skip(end)
                    if pos >= len(text) or text[pos] == "]":
                        return found
                    if text[pos] != ",":
                        return []
                    pos += 1
            _, pos = decoder.raw_decode(text, pos)
            pos = skip(pos)
            if pos >= len(text) or text[pos] != ",":
                return []
            pos += 1
    except (ValueError, RecursionError):
        return []
    return []


def recover_final(response, seen, paper_mode=True):
    from deep_research import _render_final, WorkflowStop
    if response.get("parse_error") != "output_token_limit" or response.get("output_budget_exceeded"):
        return None, None
    candidates = complete_items(response.get("raw"))
    accepted, rejected = [], []
    for index, item in enumerate(candidates):
        try:
            _render_final({"items": [item]}, seen, paper_mode)
        except (WorkflowStop, TypeError, ValueError, RecursionError) as exc:
            rejected.append({"index": index, "reason": getattr(exc, "reason", "invalid_item")})
        else:
            accepted.append(item)
    report = {"reason": "output_token_limit", "complete_items": len(candidates),
              "retained_items": len(accepted), "rejected_items": rejected,
              "extra_model_calls": 0, "semantic_accuracy_proven": False}
    if not accepted:
        return None, report
    return {"items": accepted, "unresolved": []}, report
