"""Validate a source-bound reading brief; this does not verify its semantics."""

from quality_review import locate_quote


def compact_source_context(evidence):
    """Request-local aliases shorten output without changing source text."""
    aliases = {f"s{i}": row["id"] for i, row in enumerate(evidence, 1)}
    context = "\n\n".join(
        f"[s{i} | {row['section']} | p.{row['page_start']}-{row['page_end']}]\n{row['text']}"
        for i, row in enumerate(evidence, 1)
    )
    return context, aliases


def validate_compact_structure(data, evidence, aliases):
    """Validate compact transport and resolve every alias back to this request."""
    report = {"schema": "brief_v2", "accepted_rows": 0, "decisions": [],
              "semantic_accuracy_proven": False}
    limits = {"evidence": 10, "components": 8, "relations": 6,
              "experiments": 2, "limitations": 1, "unconfirmed": 3}
    if (data.get("paper_type") not in ("method", "review", "theory", "other")
            or any(not isinstance(data.get(k), list) or len(data[k]) > n for k, n in limits.items())):
        return None, {**report, "reason": "invalid_structure_schema"}
    seen = {row["id"]: row for row in evidence}
    sources = {}
    for i, row in enumerate(data["evidence"]):
        decision = {"path": f"evidence[{i}]", "accepted": False, "reason": "invalid_schema"}
        report["decisions"].append(decision)
        if not isinstance(row, dict):
            continue
        key, ref, quote = row.get("key"), row.get("ref"), row.get("quote")
        if not all(isinstance(x, str) and x.strip() for x in (key, ref, quote)) or len(key) > 8 or len(quote) > 160:
            continue
        if key in sources:
            return None, {**report, "reason": "duplicate_evidence_key"}
        cid = aliases.get(ref)
        if cid not in seen:
            decision["reason"] = "evidence_out_of_scope"
            continue
        location = locate_quote(seen[cid]["text"], quote)
        if location is None:
            decision["reason"] = "quote_mismatch"
            continue
        sources[key] = {"id": cid, "quote": quote, "location": location,
                        "section": seen[cid].get("section"), "page_start": seen[cid].get("page_start")}
        decision.update(accepted=True, reason="source_text_matched", source_id=cid, source_location=location)
    result = {"schema": "brief_v2", "paper_type": data["paper_type"], "evidence": sources}
    names = set()
    for group in ("components", "relations", "experiments", "limitations"):
        result[group] = []
        for i, row in enumerate(data[group]):
            decision = {"path": f"{group}[{i}]", "accepted": False, "reason": "invalid_schema"}
            report["decisions"].append(decision)
            fields = {"components": {"name": 60, "stage": 20, "io": 100},
                      "relations": {"from": 60, "to": 60},
                      "experiments": {"text": 160}, "limitations": {"text": 100}}[group]
            if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip()
                                               or len(row[k]) > cap for k, cap in fields.items()):
                continue
            basis = row.get("basis")
            if not isinstance(basis, list) or not 1 <= len(basis) <= 2 or not all(isinstance(k, str) and k in sources for k in basis):
                decision["reason"] = "unverified_evidence_reference"
                continue
            if group == "components":
                if row["stage"] not in {"preprocess", "representation", "backbone", "training", "inference", "output", "engineering", "other"}:
                    continue
                if row["name"] in names:
                    return None, {**report, "reason": "duplicate_component_name"}
                names.add(row["name"])
            if group == "relations" and (row["from"] not in names or row["to"] not in names):
                decision["reason"] = "component_not_found"
                continue
            result[group].append({k: row[k] for k in fields} | {"basis": basis})
            report["accepted_rows"] += 1
            decision.update(accepted=True, reason="source_reference_matched_not_semantically_verified")
    unknown = data["unconfirmed"]
    if any(not isinstance(x, str) or not x.strip() or len(x) > 80 for x in unknown):
        return None, {**report, "reason": "invalid_structure_schema"}
    result["unconfirmed"] = unknown
    if not report["accepted_rows"] or (data["paper_type"] == "method" and not result["components"]):
        return None, {**report, "reason": "no_valid_structure"}
    return result, {**report, "reason": "structure_available_not_semantically_verified"}


LIMITS = {"method_components": 12, "method_flow": 12, "experiment_evidence": 6,
          "author_limitations": 4, "section_coverage": 16}
FIELDS = {
    "method_components": ("name", "role", "stage", "input", "output", "function"),
    "method_flow": ("from", "to", "relationship"),
    "experiment_evidence": ("comparison", "metric", "unit", "values", "conditions", "claim"),
    "author_limitations": ("text",),
    "section_coverage": ("name", "status"),
}


def validate_structure(data, evidence, aliases=None):
    if isinstance(data, dict) and data.get("schema") == "brief_v2":
        return validate_compact_structure(data, evidence, aliases or {})
    seen = {row["id"]: row["text"] for row in evidence}
    report = {"accepted_rows": 0, "decisions": [], "semantic_accuracy_proven": False}
    if (not isinstance(data, dict) or not isinstance(data.get("paper_type"), str)
            or data["paper_type"] not in {"method", "review", "theory", "other"}):
        return None, {**report, "reason": "invalid_structure_schema"}
    result = {"paper_type": data["paper_type"]}
    for name, limit in LIMITS.items():
        rows = data.get(name)
        if not isinstance(rows, list) or len(rows) > limit:
            return None, {**report, "reason": "invalid_structure_schema"}
        result[name] = []
        for index, row in enumerate(rows):
            decision = {"path": f"{name}[{index}]", "accepted": False, "reason": "invalid_schema"}
            report["decisions"].append(decision)
            if not isinstance(row, dict) or any(not isinstance(row.get(field), str)
                    or not row[field].strip() or len(row[field]) > 700 for field in FIELDS[name]):
                continue
            if name == "method_components" and (row["role"] not in {"core", "engineering"}
                    or row["stage"] not in {"preprocess", "representation", "backbone", "objective", "inference", "output", "other"}):
                continue
            if name == "section_coverage" and row["status"] not in {"read", "unconfirmed"}:
                continue
            ids, quote = row.get("evidence_ids"), row.get("source_quote")
            if not isinstance(ids, list) or not all(isinstance(cid, str) for cid in ids) or not isinstance(quote, str) or len(quote) > 500:
                continue
            if any(cid not in seen for cid in ids):
                decision["reason"] = "evidence_out_of_scope"
                continue
            if name == "section_coverage" and row["status"] == "unconfirmed" and not ids and not quote:
                result[name].append({field: row[field] for field in FIELDS[name]} | {"evidence_ids": [], "source_quote": ""})
                decision.update(accepted=True, reason="explicitly_unconfirmed")
                continue
            if not ids or not quote.strip():
                decision["reason"] = "source_not_provided"
                continue
            match = next(((cid, locate_quote(seen[cid], quote)) for cid in ids
                          if locate_quote(seen[cid], quote)), None)
            if not match:
                decision["reason"] = "quote_mismatch"
                continue
            if name == "method_flow":
                components = {item["name"] for item in result["method_components"]}
                if row["from"] not in components or row["to"] not in components:
                    decision["reason"] = "component_not_found"
                    continue
            cleaned = {field: row[field] for field in FIELDS[name]}
            cleaned.update(evidence_ids=ids, source_quote=quote)
            result[name].append(cleaned)
            report["accepted_rows"] += 1
            decision.update(accepted=True, reason="source_text_matched", source_id=match[0], source_location=match[1])
    if not report["accepted_rows"] or (data["paper_type"] == "method" and not result["method_components"]):
        return None, {**report, "reason": "no_valid_structure"}
    report["reason"] = "structure_available_not_semantically_verified"
    return result, report
