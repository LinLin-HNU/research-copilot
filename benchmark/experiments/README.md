# V1 评测实验工具

这些是评测工具，不是产品 V2/V3。`v2_experimental` 目前只是 `v1_current` 的别名，不能把两者当两个不同算法。

## 2026 10 检索消融框架

消融实验在独立的 `experiment/ablation-20261005` Worktree 中开发。实验代码优先位于
`benchmark/experiments/` 和 `tests/`，获胜方案经过确认前不修改生产检索入口。

阶段 1 先运行完全离线的协议 smoke test。它只使用合成题目、合成 chunk 和固定排名，
不会读取 `.env`、不会访问 Chroma、不会调用 embedding、路由或答案模型。其输出只能证明：

- 题目 ID 和 split 校验有效；
- R0 与 R1 使用完全相同的候选池、块数、单块字符数和总字符预算；
- 题库哈希、语料指纹、变体和预算会进入 manifest；
- 成对变体缺失或预算漂移时实验会直接失败；
- 合成 smoke 结果不会被当成项目质量指标。

```powershell
$py = 'C:\Users\86195\PycharmProjects\PythonProject2\.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
& $py -m unittest tests.test_ablation_protocol -v
& $py benchmark/experiments/smoke_ablation_protocol.py --output benchmark/experiments/private/ablation_protocol_smoke
```

输出目录必须不存在。运行成功后会生成 `manifest.json`、`retrieval_rows.jsonl` 和
`summary.json`。这些文件位于私人实验结果目录，不应提交到 Git。

### R0 与 R1 正式检索消融

`run_retrieval_ablation.py` 比较：

- R0：同一 Top 20 候选池中按向量距离取前 8 块；
- R1：同一 Top 20 候选池中用 MMR 选择 8 块。

二者都只使用用户原问题，不调用路由、查询扩展、结构补充或答案模型。每块最多
600 字，总证据最多 4000 字。脚本先复制 Chroma 再读取，不打开在线向量库。

除总命中率外，每题还记录关键词在哪一层丢失：全语料、候选池、选块，或最终预算
和截断。报告同时列出 `miss -> hit` 与 `hit -> miss`，并按题型和论文聚合。

这三个命中值是自动“代理指标”，不是语义正确率：`section_hit` 只比较解析器给出的
章节标签，`term_hit` 只检查最终证据中是否包含所有预标注字符串，`retrieval_hit`
是二者同时命中。它们适合做可重复的筛查和失败归因，但同义改写可造成假阴性，无关上下文、
否定句或参考文献中的词又可造成假阳性。所以报告行中额外写入了
`section_label_hit`、`lexical_term_hit` 和 `strict_proxy_hit` 这组更不容易被误解的别名。

真实运行会把问题文本发送到项目 `.env` 配置的 embedding API。72 题由现有批处理
最多分成约 4 个 HTTP 请求；不会发送论文 chunk，不调用路由或答案生成模型。

## 当前交付状态

- 切窗、top-k、盲评、Wilson 区间、一致率和实验控制流通过离线验证。
- 真实 API 试跑被自动审批拦截，尚无本次的真实基线答案或成本对照。
- 已有 30 题全部作为开发题，未伪造 70 道测试题或独立人工标签。
- 原题 `benchmark/questions.json` 和线上检索/提示词保持不变。
- `private/` 下保存本地论文快照、题目、答案、标注和报告；Git/Docker 均排除。不要提交或分享原始文件。

## 运行环境

在项目根目录运行。建议使用项目自己的虚拟环境：

```powershell
$py = '.\.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
```

## 先核对标准，再生成答案

打开 `private/audit_verified/gold_annotation.csv`。候选原文是按原有关键词找到的帮助材料，不是独立标准答案。
填写 reference_key_points、verified_citations、reviewer、approved。对照 PDF 物理页核对。不要为适应结果修改 expected_*。
把核对通过的要点写回 `private/audit_verified/dev_cases.json` 的 gold；保留原始题库不动。

## 小规模真实试跑（需同意外部 API 数据传输及费用）

以下命令会发送题目和论文片段到 `.env` 中配置的 DashScope 兼容接口。输出目录不能已存在，以防不同轮结果混合。

```powershell
& $py benchmark/experiments/run_experiment.py --cases benchmark/experiments/private/audit_verified/dev_cases.json --variant v0_naive --split dev --limit 2 --output benchmark/experiments/private/smoke_v0
& $py benchmark/experiments/run_experiment.py --cases benchmark/experiments/private/audit_verified/dev_cases.json --variant v1_current --split dev --limit 2 --output benchmark/experiments/private/smoke_v1
```

v0：已有块按页序拼接，1000 字符固定窗、100 字符重叠、纯余弦 top-5、普通上下文提示词。
它不是从原始 PDF 重新解析的完美基线：源块可能已有重叠，切窗会再次重复。报告必须注明。

v1：调用现有 build_route / build_question_evidence / request_prompt，用同一个模型和现有动态提示词中间件。采用独立单轮、无持久化 checkpoint 的 Agent，避免污染日常会话。它覆盖问答核心链路，不测 HTTP/SSE、多轮历史、上传摘要全流程。

v0/v1 同时改变了切块、路由、选块和提示词，所以这是系统级对照，不是可以归因到 MMR 的消融实验。B 类在 v0 中可能因“上下文没有就不知道”而拒答，要按题型分层解释。

每题落盘，记录真实耗时、最终消息 usage、路由 usage 和 embedding API 返回的 usage；缺失为 null，不估算账单。v0 首题包含建索引费用，后续命中内存缓存；不要将它与 v1 热查询直接作为成本下降结论。模型失败重试可能额外收费，回调记录不等于完整账单。

`manifest.json` 记录代码和题库 SHA256，`corpus.json` 记录语料块数量/哈希。实验使用私有目录下的 Chroma 副本，避免修改 live 集合。为保证副本一致，复制时不要上传/删除论文；快照保留以便复查，会额外占磁盘空间。

## 全量运行统计与盲评

```powershell
& $py benchmark/experiments/summarize_runs.py benchmark/experiments/private/smoke_v0/answers_v0_naive.jsonl benchmark/experiments/private/smoke_v1/answers_v1_current.jsonl --output benchmark/experiments/private/run_summary.json
& $py benchmark/experiments/make_blind_sheet.py benchmark/experiments/private/smoke_v0/answers_v0_naive.jsonl benchmark/experiments/private/smoke_v1/answers_v1_current.jsonl --output benchmark/experiments/private/blind
& $py benchmark/experiments/score_blind_sheet.py --review benchmark/experiments/private/blind/review.csv --mapping benchmark/experiments/private/blind/mapping.private.json --output benchmark/experiments/private/human_scores.json
```

`summarize_runs` 统计所有尝试，包括失败。`make_blind_sheet` 仅收录各变体都成功的完整配对；被排除的失败不能隐藏，应同时报告 run_summary。
评分期间不打开 mapping.private.json。顺序随机、去掉显式版本标签，但文风可能暴露来源，不能保证绝对盲法。
评分列只接受 `0`、`1`、`NA` 或空白。空白/NA 不计入分母，不能当成 0。四项规范：

| 列 | 1 | 0 | NA |
|---|---|---|---|
| citation_correct | 应引用的论文事实引用真实且支持结论 | 存在错引或该引未引 | 纯通用知识无需论文引用 |
| grounded | 论文事实由材料支持，外部知识/推断有明确区分 | 有未经支持却当作论文事实的陈述 | 无可判定论文主张 |
| hallucination | 至少一处捏造或无依据事实 | 未发现 | 材料不足以判断 |
| completeness | 覆盖事先核定的必答要点 | 缺少至少一个必答要点 | 参考要点尚未核定 |

注意 hallucination 越低越好，其他三项越高越好。二元标准较粗，notes 中解释原因。
第二评阅人使用同一份 review_id 表独立填写，通过 `--second-review 文件.csv` 计算一致率和 Cohen's kappa。所有人都给同一分导致 kappa 无定义时输出 null。
表中显示的差值是同题配对百分点差，不包含差值置信区间或显著性检验；各比例的区间是 Wilson 95%。题目按论文聚集，独立伯努利假设只是近似，不能把区间当作全领域泛化保证。

## 冻结测试纪律

已用过的 30 题全部是 dev。新增题目必须先确定 gold 和来源，建议用不同论文留出测试语料，减少泄漏。
测试题每条带 split=test。人工核定后创建 manifest，含 cases_sha256（题库文件哈希）和 human_gold_approved=true。
运行时提供 `--split test --test-manifest ... --variant all --confirm-full`，同一次跑完要比较的变体。
首次正式运行会建立同名 `.started` 标记，再运行会拒绝；失败如实报告，不能悄悄删除标记刷分。此保护是工作流约束，不是防篡改安全机制。

## 验证

```powershell
& $py -m unittest discover -s tests -v
& $py benchmark/experiments/verify_delivery.py --output benchmark/experiments/private/verification_new.json
```

离线测试中的 Synthetic 和 token 11/7 都是模拟夹具，仅证明程序行为，不属于论文效果或真实 token 测量。

## 路由专项复测（不生成答案）

`evaluate_routes.py` 只调用廉价的路由分类器，不检索、不生成答案。
可用旧 V1 JSONL 作修改前基线，报告总准确率、各类召回率和非 B 题被误判 B 的数量。

```powershell
& $py -X faulthandler -u benchmark/experiments/evaluate_routes.py `
  --cases benchmark/drafts/timeseries_20260922_curated_release2/ai_reviewed_dev_cases.json `
  --split dev `
  --baseline-answers benchmark/experiments/private/ai_gold_full_v1/answers_v1_current.jsonl `
  --confirm-full `
  --output benchmark/experiments/private/route_after_guard.json
```

该脚本会调用路由 API，但不会调用答案生成 API。运行前确认 `.env` 的模型配置可用。

## 相同 chunk / 相同预算的检索对照

`compare_retrieval_fair.py` 不再把 V0 重新切成 1000 字窗口。纯向量与 MMR 都从
同一份 Chroma 原始 chunk 中选择，统一为每块最多 600 字、总证据最多 4000 字、
最多 8 块。它只隔离比较“纯向量排名 vs MMR”，不是完整 V0/V1 系统效果。

```powershell
& $py -X faulthandler -u benchmark/experiments/compare_retrieval_fair.py `
  --cases benchmark/drafts/timeseries_20260922_curated_release2/ai_reviewed_dev_cases.json `
  --split dev `
  --confirm-full `
  --max-chunk-chars 600 `
  --char-budget 4000 `
  --output benchmark/experiments/private/fair_retrieval_600_4000
```

该脚本会调用 embedding API，不调用路由和答案生成 API；会复制 Chroma 快照，
不修改在用向量库。

## LLM-as-a-Judge（不是人工盲审）

先做 2 行、1 轮的付费试跑：

```powershell
& $py -X faulthandler -u benchmark/experiments/run_llm_judge.py `
  --review-packets benchmark/experiments/private/ai_gold_blind_review/review_source_packets.csv `
  --passes 1 `
  --limit 2 `
  --output benchmark/experiments/private/llm_judge_smoke
```

试跑成功后再跑 144 行、2 轮（共 288 次评判调用）：

```powershell
& $py -X faulthandler -u benchmark/experiments/run_llm_judge.py `
  --review-packets benchmark/experiments/private/ai_gold_blind_review/review_source_packets.csv `
  --passes 2 `
  --confirm-full `
  --output benchmark/experiments/private/llm_judge_full

& $py benchmark/experiments/score_llm_judge.py `
  --judgements benchmark/experiments/private/llm_judge_full/judgements.jsonl `
  --mapping benchmark/experiments/private/ai_gold_blind_review/mapping.private.json `
  --output benchmark/experiments/private/llm_judge_scored.json
```

评判阶段不读 `mapping.private.json`，模型不知道变体身份；只在评判全部落盘后由
`score_llm_judge.py` 揭盲汇总。两轮不一致或任一轮证据不足的单元格按 `NA` 处理。
报告字段明确为 `llm_as_a_judge_not_human_review`，不会写入 `review.csv`。
