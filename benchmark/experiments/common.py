"""Dependency-free evaluation primitives. No project imports or network calls."""
import hashlib
import json
import math


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def windows(chunks, size=1000, overlap=100):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > overlap >= 0")
    ordered = sorted(chunks, key=lambda r: (r.get("page_start") or 0, r.get("_order", 0)))
    text, spans = "", []
    for chunk in ordered:
        start = len(text)
        text += chunk["content"] + "\n"
        spans.append((start, len(text), chunk))
    result = []
    for start in range(0, len(text), size - overlap):
        end = min(start + size, len(text))
        sources = [r for a, b, r in spans if a < end and b > start]
        result.append({"content": text[start:end], "start": start, "end": end,
                       "sections": sorted({r.get("section", "unknown") for r in sources}),
                       "source_ids": [r["id"] for r in sources],
                       "page_start": min((r.get("page_start") or 0 for r in sources), default=0),
                       "page_end": max((r.get("page_end") or 0 for r in sources), default=0)})
        if end == len(text):
            break
    return result


def top_k(query, vectors, k):
    if k <= 0 or any(len(v) != len(query) for v in vectors):
        raise ValueError("Positive k and matching vector dimensions required")
    def cosine(v):
        denominator = math.sqrt(sum(x*x for x in query) * sum(x*x for x in v))
        return sum(a*b for a, b in zip(query, v)) / denominator if denominator else 0.0
    return sorted(range(len(vectors)), key=lambda i: (-cosine(vectors[i]), i))[:k]


def retrieval_metrics(case, evidence, sections):
    # Eligibility is based on GOLD, never the router's prediction.
    if case["expected_category"] == "B":
        return {"retrieval_hit": None, "section_hit": None, "term_hit": None}
    wanted = {s.lower() for s in case.get("expected_sections", [])}
    terms = case.get("expected_terms", [])
    if not wanted and not terms:
        return {"retrieval_hit": None, "section_hit": None, "term_hit": None}
    section_hit = not wanted or bool(wanted & {s.lower() for s in sections})
    term_hit = all(term.lower() in evidence.lower() for term in terms)
    return {"retrieval_hit": bool(evidence) and section_hit and term_hit,
            "section_hit": section_hit, "term_hit": term_hit}


def pack_chunks(chunks, char_budget=4000, max_chunk_chars=600, max_chunks=8):
    """Format ranked *original* chunks under one exact, shared evidence budget.

    This deliberately does not re-window text.  It is used by the fair retrieval
    comparison so vector Top-K and MMR see the same Chroma chunk ids and receive
    the same per-chunk and total-character limits.
    """
    if char_budget <= 0 or max_chunk_chars <= 0 or max_chunks <= 0:
        raise ValueError("Budgets and limits must be positive")
    separator = "\n\n---\n\n"
    parts, selected, seen = [], [], set()
    for chunk in chunks:
        identity = chunk.get("id") or chunk.get("content")
        if identity in seen:
            continue
        content = (chunk.get("content") or "")[:max_chunk_chars]
        if not content.strip():
            continue
        section = chunk.get("section") or "unknown"
        page_start = chunk.get("page_start", 0) or 0
        page_end = chunk.get("page_end", 0) or 0
        page = f" | 第 {page_start}-{page_end} 页" if page_end else ""
        part = f"【来源: {section} | chunk_id: {chunk.get('id', 'unknown')}{page}】\n{content}"
        candidate = separator.join([*parts, part])
        if len(candidate) > char_budget:
            break
        parts.append(part)
        selected.append({**chunk, "content_emitted": content})
        seen.add(identity)
        if len(parts) >= max_chunks:
            break
    return separator.join(parts), selected


def proportion(values):
    values = [v for v in values if v is not None]
    n, successes = len(values), sum(bool(v) for v in values)
    if not n:
        return {"n": 0, "successes": 0, "rate": None, "ci95": None}
    p, z = successes / n, 1.959963984540054
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return {"n": n, "successes": successes, "rate": p,
            "ci95": [max(0, center-radius), min(1, center+radius)]}


def agreement(pairs):
    if not pairs:
        return {"n": 0, "agreement": None, "kappa": None}
    n = len(pairs)
    observed = sum(a == b for a, b in pairs)/n
    p1, p2 = sum(a for a, _ in pairs)/n, sum(b for _, b in pairs)/n
    expected = p1*p2 + (1-p1)*(1-p2)
    return {"n": n, "agreement": observed,
            "kappa": (observed-expected)/(1-expected) if expected < 1 else None}
