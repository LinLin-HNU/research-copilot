"""Summarize ALL attempted cases, including API failures; no human quality claims."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from common import read_jsonl, proportion


def numeric(values):
    known = sorted(v for v in values if v is not None)
    return {"observed_n":len(known),"missing_n":len(values)-len(known),
            "mean":sum(known)/len(known) if known else None,
            "p50":known[math.ceil(.5*len(known))-1] if known else None,
            "p95":known[math.ceil(.95*len(known))-1] if known else None}


def summary(rows):
    eligible = [r for r in rows if r['expected_category'] != 'B' and (r.get('expected_terms') or r.get('expected_sections'))]
    return {"attempted_n":len(rows), "successful_answers":proportion([not r.get('error') and bool(r.get('answer')) for r in rows]),
            "route_accuracy_observed":proportion([r.get('route_correct') for r in rows]),
            "route_unobserved_n":sum(r.get('route_correct') is None for r in rows),
            "retrieval_hit_all_eligible":proportion([bool(r.get('retrieval_hit')) for r in eligible]),
            "retrieval_unobserved_n":sum(r.get('retrieval_hit') is None for r in eligible),
            "generation_input_tokens":numeric([r.get('input_tokens') for r in rows]),
            "generation_output_tokens":numeric([r.get('output_tokens') for r in rows]),
            "routing_input_tokens":numeric([r.get('routing_usage',{}).get('input_tokens') for r in rows]),
            "routing_output_tokens":numeric([r.get('routing_usage',{}).get('output_tokens') for r in rows]),
            "stage_ms":{stage:numeric([r.get('timing_ms',{}).get(stage) for r in rows]) for stage in ('index','routing','retrieval','generation','total')}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('answers',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=[r for path in args.answers for r in read_jsonl(path)]
    if len({(r['variant'],r['id']) for r in rows}) != len(rows):
        p.error('Duplicate case/variant: summarize one run only')
    result={'limitations':['No currency costs; missing usage is unknown, not zero','v0 index cost appears only on first question per paper; compare separately from warm queries','Not answer-quality metrics; human review required'], 'overall':{},'by_category':{},'by_paper':{},'failures':[]}
    for variant in sorted({r['variant'] for r in rows}):
        subset=[r for r in rows if r['variant']==variant]
        result['overall'][variant]=summary(subset)
        for field,target in (('expected_category','by_category'),('thread_id','by_paper')):
            groups=defaultdict(list)
            for r in subset:
                groups[r[field]].append(r)
            result[target][variant]={k:summary(v) for k,v in groups.items()}
    for r in rows:
        if r.get('error') or r.get('retrieval_hit') is False or r.get('route_correct') is False:
            result['failures'].append({k:r.get(k) for k in ('id','variant','error','route_correct','retrieval_hit','section_hit','term_hit')})
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result['overall'],ensure_ascii=False))


if __name__=='__main__':
    main()
