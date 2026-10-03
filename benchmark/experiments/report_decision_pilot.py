"""Create an offline, human-readable report for the decision pilot."""
import argparse
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path

VARS = ('v0_naive', 'v1_current', 'v1_simple')
NAMES = {'v0_naive':'V0 朴素基线', 'v1_current':'当前 V1', 'v1_simple':'简化实验版'}

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def mean(values): return round(statistics.mean(values), 1) if values else None
def pct(a,b): return f'{a}/{b}（{100*a/b:.1f}%）' if b else '不适用'
def esc(value): return html.escape(str(value or ''))

def summarize(rows):
    result = {}
    for variant in VARS:
        subset = [r for r in rows if r['variant']==variant]
        paper = [r for r in subset if r['input_mode']=='paper']
        general = [r for r in subset if r['input_mode']=='general']
        def tokens(items, key): return mean([r.get('usage',{}).get(key) for r in items if r.get('usage',{}).get(key) is not None])
        result[variant] = {
            'all_n':len(subset), 'errors':sum(bool(r.get('error')) for r in subset),
            'length_truncated':sum(r.get('response_metadata',{}).get('finish_reason')=='length' for r in subset),
            'paper_n':len(paper), 'paper_retrieval_hits':sum(r.get('retrieval_hit') is True for r in paper),
            'paper_section_hits':sum(r.get('section_hit') is True for r in paper),
            'paper_term_hits':sum(r.get('term_hit') is True for r in paper),
            'mean_evidence_chars_paper':mean([r.get('evidence_chars',0) for r in paper]),
            'mean_input_tokens_all':tokens(subset,'input_tokens'), 'mean_output_tokens_all':tokens(subset,'output_tokens'),
            'mean_input_tokens_paper':tokens(paper,'input_tokens'), 'mean_input_tokens_general':tokens(general,'input_tokens'),
            'mean_total_ms_all':mean([r['timing_ms']['total'] for r in subset]),
            'mean_total_ms_paper':mean([r['timing_ms']['total'] for r in paper]),
            'mean_total_ms_general':mean([r['timing_ms']['total'] for r in general]),
        }
        if variant=='v1_current':
            result[variant]['route_accuracy_paper'] = sum(r.get('route',{}).get('category')==r['expected_category'] for r in paper)
            result[variant]['route_accuracy_general'] = sum(r.get('route',{}).get('category')=='B' for r in general)
            result[variant]['mean_routing_input_tokens'] = mean([r.get('routing_usage',{}).get('input_tokens') for r in subset if r.get('routing_usage',{}).get('input_tokens') is not None])
    return result

def make_report(run_dir, calibration, output):
    rows = [json.loads(x) for x in (run_dir/'answers.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    manifest, corpus, cal = read_json(run_dir/'manifest.json'), read_json(run_dir/'corpus.json'), read_json(calibration)
    if len(rows)!=54 or {(r['id'],r['variant']) for r in rows} != {(r['id'],v) for r in rows for v in VARS}:
        raise ValueError('Expected exactly 18 complete three-variant cases')
    stats = summarize(rows)
    output.mkdir(parents=True,exist_ok=False)
    summary = {'protocol':manifest['protocol'],'model':manifest['model'],'records':len(rows),'case_count':18,
               'paper_case_count':12,'general_case_count':6,'stats':stats,
               'judge_calibration':{k:cal[k] for k in ('passed_checks','metric_checks','critical_faithfulness_passed','ready_for_small_comparison','by_metric')},
               'quality_winner':None,
               'quality_winner_reason':'裁判合成验收未全部通过，且没有人工论文核验；不从过程命中率推出答案质量赢家。',
               'limitations':manifest['limitations']}
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

    table=[]
    for v in VARS:
        s=stats[v]
        route = '不使用' if v!='v1_current' else f"论文 {pct(s['route_accuracy_paper'],s['paper_n'])}；通用题 {pct(s['route_accuracy_general'],6)}"
        table.append(f"| {NAMES[v]} | {pct(s['paper_retrieval_hits'],s['paper_n'])} | {pct(s['paper_section_hits'],s['paper_n'])} | {pct(s['paper_term_hits'],s['paper_n'])} | {s['mean_evidence_chars_paper']} | {s['mean_input_tokens_all']} / {s['mean_output_tokens_all']} | {s['mean_total_ms_all']} | {route} |")
    md=f"""# 科研助手方向决策小实验（2026-09-30）

## 先看结论

本轮同一个模型 `{manifest['model']}` 共生成 54 个回答（18 题 × 3 方案），全部成功，0 个截断。12 道论文题覆盖 9 篇论文；另有 6 道通用知识题。它是开发集诊断，不是独立测试，也没有比较 ChatGPT 直接上传整篇 PDF。

**目前不能宣布质量赢家。** 自动裁判先做了 8 个合成案例、24 个固定检查项，只通过 {cal['passed_checks']}/{cal['metric_checks']}；忠实性 8/8，但引用 6/8、完整性 7/8。因此不让未校准好的裁判给真实论文答案排名。

| 方案 | 严格联合命中 | 章节命中 | 关键词全命中 | 论文证据均长(字符) | 全题平均输入/输出token | 全题平均耗时(ms) | 路由 |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(table)}

## 这些数字怎么理解

- `严格联合命中` 要求预设章节和全部预设关键词同时出现。它适合发现漏检，不等于答案正确率。V0 的 1000 字滑窗更容易“碰到”关键词；简化版保留完整原始块、平均证据更长，却只有 {stats['v1_simple']['paper_retrieval_hits']}/12，说明纯向量 Top-K 仍会把正确实体或关键句排在预算之外。
- 当前 V1 的 12 道论文题路由为 {pct(stats['v1_current']['route_accuracy_paper'],12)}，6 道通用题判成 B 为 {pct(stats['v1_current']['route_accuracy_general'],6)}。路由准确不代表后续检索必然完整。
- 简化版的价值不是已证明“更准确”，而是把产品假设变得可检验：用户明确选择“论文内回答/通用知识”，论文模式不再依赖 A1/A2/C 分类；失败更容易归因到检索本身。
- 耗时含本机索引缓存、网络和调用顺序影响，不是线上吞吐基准。输入 token 未把嵌入计费统一折算进去；当前 V1 的路由输入 token 另有记录。

## 裁判为什么没通过

- `cal_03` 的答案没有引用，裁判却声称有章节和页码；同一题完整性理由虽正确，却生成了证据中不存在的引文短语，被校验器拦下。
- `cal_04` 的证据刻意删除页码，裁判仍说答案页码与证据匹配。
- 这证明结构化 JSON 和同模型打分并不自动可靠。裁判输出保留作失败证据，但不用于本轮真实论文质量排名。

## 对开发方向的建议

先不要继续堆 GraphRAG 或多 Agent。下一版把产品定位为“可核验的科研阅读工作台”，RAG 是其中一个工具，而不是项目名称本身：

1. 保留显式的论文内/通用知识模式，弱化 A1/A2/C 对检索范围的硬控制；A1/A2/C 可留作回答模板或观测标签。
2. 下一项实验不是再改评分尺，而是做 12 题的小规模证据标注：只标出支持答案的 chunk id。之后可直接测 Recall@k/MRR，绕开大窗口碰词的偏差。
3. 加一个“整篇输入”基线（同一模型、同一 12 题、相同输出预算）。只有证明 RAG 在成本、延迟或引用定位上有取舍优势，才有简历上可信的工程故事。
4. GraphRAG 只在跨论文全局问题（方法演化、共识/冲突、实体关系）出现明确需求后试验；它不是单篇事实检索的自然升级。多 Agent 也应服务于检索—核验—引用这一条业务闭环。

## 实验边界

- 三方案同时改变了路由、切块/选块和提示词，是系统比较，不是单变量消融。
- 简化版读取的是用户显式 `paper/general` 模式，不是利用标准答案作弊，也不能声称自动路由提升。
- 参考要点是 AI-reviewed 开发材料，不是人工金标准；本轮没有人工读论文验真。
- HTML 中保留每题三个答案和各自实际证据，供你直观看差异；不要把本页命中率写成“回答准确率”。
"""
    (output/'report.md').write_text(md,encoding='utf-8')

    by_id=defaultdict(dict)
    for r in rows: by_id[r['id']][r['variant']]=r
    style="""body{font-family:Segoe UI,Microsoft YaHei,sans-serif;max-width:1500px;margin:30px auto;padding:0 22px;color:#17202a;background:#f6f8fb}h1,h2{color:#153e75}.notice{padding:16px;border-left:5px solid #d69e2e;background:#fffaf0}.good{border-left-color:#2f855a;background:#f0fff4}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{background:white;border:1px solid #d8dee9;border-radius:10px;padding:14px;min-width:0}.answer,.evidence{white-space:pre-wrap;word-break:break-word;font-size:14px}.answer{background:#f8fafc;padding:10px}.meta{color:#4a5568;font-size:13px}details{margin-top:10px}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border:1px solid #cbd5e0;text-align:left}section{margin:30px 0}@media(max-width:1000px){.cards{grid-template-columns:1fr}}"""
    trs=''.join(f"<tr><td>{esc(NAMES[v])}</td><td>{pct(stats[v]['paper_retrieval_hits'],12)}</td><td>{pct(stats[v]['paper_section_hits'],12)}</td><td>{pct(stats[v]['paper_term_hits'],12)}</td><td>{stats[v]['mean_evidence_chars_paper']}</td><td>{stats[v]['mean_input_tokens_all']} / {stats[v]['mean_output_tokens_all']}</td><td>{stats[v]['mean_total_ms_all']}</td></tr>" for v in VARS)
    sections=[]
    for cid in sorted(by_id,key=lambda x:(not x.startswith('general'),x)):
        cards=[]
        for v in VARS:
            r=by_id[cid][v]; u=r.get('usage',{})
            route=r.get('route') or {}; hit='不适用' if r.get('retrieval_hit') is None else ('命中' if r['retrieval_hit'] else '未命中')
            cards.append(f"<article class='card'><h3>{esc(NAMES[v])}</h3><p class='meta'>联合命中：{hit}；证据 {r.get('evidence_chars',0)} 字；输入/输出 {u.get('input_tokens','?')}/{u.get('output_tokens','?')} token；路由 {esc(route.get('category','—'))}</p><div class='answer'>{esc(r['answer'])}</div><details><summary>查看实际送给模型的证据</summary><div class='evidence'>{esc(r['evidence']) or '（无论文证据）'}</div></details></article>")
        sections.append(f"<section><h2>{esc(cid)}　{esc(by_id[cid][VARS[0]]['question'])}</h2><div class='cards'>{''.join(cards)}</div></section>")
    page=f"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>科研助手方向决策小实验</title><style>{style}</style><body><h1>科研助手方向决策小实验</h1><div class='notice'><b>不能宣布质量赢家。</b> 54/54 回答成功，但自动裁判合成验收仅 {cal['passed_checks']}/{cal['metric_checks']}，故未给真实论文答案自动排名。过程命中率不是回答准确率。</div><p>模型：{esc(manifest['model'])}；12 道论文题 + 6 道通用题；每题三个方案；开发诊断非独立测试。</p><h2>过程数字</h2><table><tr><th>方案</th><th>严格联合命中</th><th>章节命中</th><th>关键词全命中</th><th>论文证据均长</th><th>平均输入/输出token</th><th>平均总耗时ms</th></tr>{trs}</table><div class='notice good'><b>怎么用：</b>展开任意题，先比较三个答案，再点开证据看答案是否真的有来源。简化版是“显式模式 + 纯向量完整块 + 短提示词”的实验，不是已上线 V2。</div>{''.join(sections)}</body></html>"""
    (output/'report.html').write_text(page,encoding='utf-8')
    print(json.dumps({'rows':len(rows),'output':str(output),'judge_ready':cal['ready_for_small_comparison']},ensure_ascii=False))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True); p.add_argument('--calibration',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); make_report(a.run,a.calibration,a.output)
if __name__=='__main__': main()
