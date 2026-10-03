"""Read existing reports and DB; produce reproducible evidence without API calls."""
import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from contextlib import nullcontext
from common import digest, proportion

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def summarize(rows):
    gold_eligible = [r for r in rows if r['expected_category'] != 'B']
    known = [r for r in gold_eligible if r.get('retrieval_hit') is not None]
    return {"cases": len(rows), "routing": proportion([r.get("route_correct") for r in rows]),
            "retrieval_gold_denominator": proportion([bool(r.get("retrieval_hit")) for r in gold_eligible]),
            "retrieval_observed_only": proportion([r.get("retrieval_hit") for r in known]),
            "retrieval_missing": len(gold_eligible)-len(known)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases_path = ROOT/'benchmark/questions.json'
    cases = json.loads(cases_path.read_text(encoding='utf-8'))
    by_id = {r['id']: r for r in cases}
    reports = {}
    for name in ('v1', 'v2'):
        path = ROOT/f'benchmark/reports/{name}.json'
        source = json.loads(path.read_text(encoding='utf-8'))
        rows = [{**by_id.get(r['id'], {}), **r} for r in source['results']]
        report = {"file_sha256": digest(path), "generated_at": source.get('generated_at'),
                  "original_routing_accuracy": source.get('routing_accuracy'),
                  "original_retrieval_hit_rate": source.get('retrieval_hit_rate'),
                  "overall": summarize(rows), "by_category": {}, "by_paper": {},
                  "misses": [r['id'] for r in rows if r['expected_category'] != 'B' and not r.get('retrieval_hit')],
                  "provenance_limit": "Historical report lacks source/corpus hashes; paper ids joined from current cases, not proven historical snapshot"}
        for key, target in (('expected_category','by_category'), ('thread_id','by_paper')):
            groups = defaultdict(list)
            for r in rows:
                groups[r.get(key,'unknown')].append(r)
            report[target] = {k:summarize(v) for k,v in groups.items()}
        reports[name] = report
    data = {"cases": len(cases), "papers": len({r['thread_id'] for r in cases}),
            "categories": dict(Counter(r['expected_category'] for r in cases)),
            "cases_sha256": digest(cases_path), "reports": reports}
    conn = sqlite3.connect((ROOT/'resources/research_copilot.db').as_uri()+'?mode=ro', uri=True)
    try:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute('SELECT success,total_ms,input_tokens,output_tokens FROM request_metrics')]
        data['telemetry'] = {"count": len(rows), "success": proportion([r['success'] for r in rows]),
                             "input_tokens_observed": sum(r['input_tokens'] is not None for r in rows),
                             "output_tokens_observed": sum(r['output_tokens'] is not None for r in rows)}
    finally:
        conn.close()
    (args.output/'historical_audit.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    dev = [{**c, 'split': 'dev', 'gold': {'reference_key_points': [], 'citations': [], 'status': 'needs_human_review'}} for c in cases]
    (args.output/'dev_cases.json').write_text(json.dumps(dev, ensure_ascii=False, indent=2), encoding='utf-8')
    # Suggestions are from source chunks, not generated answers; not independent gold.
    from rag_store import PaperRAG
    with nullcontext(str(args.output/'corpus_snapshot')) as temp:
        snapshot = Path(temp)/'chroma'
        shutil.copytree(ROOT/'resources/chroma_db', snapshot)
        rag = PaperRAG(str(snapshot))
        chunks = {}
        for tid in {r['thread_id'] for r in cases}:
            rag.client.get_collection(f'paper_{tid}')
            chunks[tid] = rag.list_chunks(tid)
        with (args.output/'gold_annotation.csv').open('x', encoding='utf-8-sig', newline='') as f:
            fields = ['id','question','expected_category','expected_sections','expected_terms','candidate_source_excerpts','reference_key_points','verified_citations','reviewer','approved']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for c in cases:
                terms = c.get('expected_terms', [])
                matches = [r for r in chunks[c['thread_id']] if terms and any(t.lower() in r['content'].lower() for t in terms)]
                excerpts = []
                for r in matches[:3]:
                    start = min((r['content'].lower().find(t.lower()) for t in terms if t.lower() in r['content'].lower()), default=0)
                    excerpts.append(f"{r['section']} | PDF {r['page_start']}-{r['page_end']} | {r['content'][max(0,start-150):start+550]}")
                writer.writerow({**{k:c[k] for k in ('id','question','expected_category')},
                    'expected_sections': ';'.join(c.get('expected_sections',[])), 'expected_terms': ';'.join(terms),
                    'candidate_source_excerpts': '\n\n'.join(excerpts)})
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
