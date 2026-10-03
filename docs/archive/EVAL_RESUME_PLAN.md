# 评测与简历行动清单（EVAL & RESUME PLAN）

> 这份文档一半给**我自己**看（搞清楚接下来要做什么、为什么），一半是**直接粘贴给 Codex 的提示词**（让它帮我实现）。
> 当前家底：3 篇论文、30 题，路由 100%、检索命中率 91.7%（`benchmark/reports/v2.json`）。规模偏小，下面是把它做成"简历站得住的项目"的路线。

---

# 第一部分：我要做什么（用大白话）

## 目标

把评测规模从 **3 篇 / 30 题** 扩到 **10 篇 / 100 题**，并补齐四类指标：
1. 路由准确率（机器判）
2. 检索命中率（机器判）
3. 回答质量：引用对不对、有没有根据、有没有幻觉（**必须人看**）
4. 性能与成本：延迟 p50/p95、成功率、token（真实使用积累）

为什么要扩：3 篇论文的高百分比可能是"运气好/题型窄"。10 篇覆盖不同领域、长短、标准与非标准结构后，数字才有代表性，面试也经得起追问。

## 行动步骤

| # | 做什么 | 谁来做 | 产出 |
|---|---|---|---|
| 1 | 再上传 **7 篇**论文（凑够 10）。刻意挑多样的：标准结构 + 非标准综述、不同领域（CV/NLP/时序/理论）、长短不一 | 我（网页上传） | 10 个 thread 的论文库 |
| 2 | 记录每篇的 thread_id（会话列表或 `inspect_session.py`） | 我 | thread 清单 |
| 3 | 每篇生成 **10 题草稿**（A1×5/A2×1/B×2/C×2），用 Codex 提示词 P1 自动起草 | Codex 起草 + **我核对** | `benchmark/questions.json` 扩到 ~100 题 |
| 4 | 跑确定性评测，得到稳定的路由/检索数字 | 我跑脚本 | `benchmark/reports/v*.json` |
| 5 | 让系统把 100 题**真实回答一遍并存档**（不用手动在网页逐题问） | Codex 提示词 P2 | 每题的答案 + 证据 + 来源 |
| 6 | **人工评阅**答案：引用对不对、是否接地、有没有幻觉，填 CSV | 我 | `benchmark/review.csv` → 汇总 |
| 7 | 攻检索短板（实体块落选，见复盘 P0），用同一题库做前后对照 | Codex 提示词 P3 + 我确认 | 命中率前后对比 |
| 8 | 平时多用产品对话，积累真实遥测 | 我 | `metrics_report.py` 数字 |
| 9 | 用真实数字写简历（措辞见下） | 我 | 简历条目 |

建议顺序：**1→2→3→4 先把规模做起来；5→6 拿到回答质量；7 做优化闭环；8 持续；9 最后写。**

## 关于花钱与时间的提醒

- 步骤 3（出题）只读论文块、不生成答案，便宜。
- 步骤 5 要生成 100 个真实回答，是主要花费；用 flash 模型可控。可以先做 30 题跑通流程，确认无误再跑满 100。
- 步骤 6 必须自己看，机器判幻觉不可靠——这正是评测设计的诚实之处，面试时也是加分项。

## 数字怎么变成简历（模板，数字用真实结果替换）

> 不要写没测过的数字，也不要把"字符估算"说成"账单 token"。

- 独立设计并实现证据绑定论文阅读 Agent（FastAPI + LangGraph + ChromaDB），支持章节/页码级溯源的结构化摘要与多轮问答。
- 自建 **10 篇论文 / 100 题**评测集，区分"机器判定（路由/检索）"与"人工评阅（引用/接地/幻觉）"，路由准确率 __%、检索命中率 __%、人工接地率 __%。
- 通过动态提示词注入 + 历史工具消息清理，将多轮对话 token 从随轮次线性增长控制为基本恒定。
- 定位"章节正确但关键实体块落选"的检索短板，引入 __（混合检索/精排）__ 使检索命中率从 __% 提升到 __%（同一题库前后对照）。
- 落地无内容遥测（不存问题/证据/答案）与离线回归测试，覆盖噪声清洗、证据预算与隐私边界。

---

# 第二部分：给 Codex 的提示词（直接复制）

> 用法：一次只贴一个提示词，让 Codex 做完并自测后，再进行下一个。Codex 工作在同一仓库，可运行脚本；涉及真实模型调用时需要本地 `.env` 已配好凭证。明确要求它**不要破坏现有测试、改完跑 `python -m unittest discover -s tests`**。

---

## 提示词 P1：自动出题草稿生成器

```text
我在为 Research Copilot（论文阅读助手）扩充评测题库。请帮我写一个"出题草稿生成器"脚本，不要直接改现有 questions.json。

背景与现状：
- 论文块通过 rag_store.py 的 PaperRAG.list_chunks(thread_id) 获取，每个块是 dict，含 id/content/section/page_start/page_end/_order。
- 现有题库格式见 benchmark/questions.json：数组，元素为
  {"id","question","expected_category","expected_sections","expected_terms","thread_id"}。
- 类别：A1 论文事实、A2 论文局限、B 论文外领域知识（expected_sections/terms 为空）、C 创新推理。
- 评测脚本 benchmark/run_benchmark.py 会用 expected_sections 与检索到的证据章节求交集，并用 expected_terms（小写匹配）校验证据里必须出现这些词。

请实现 benchmark/draft_questions.py：
1. 参数：--thread-id（必填）、--prefix（题目 id 前缀，如 r/s/a）、--count（默认 10）、--output（默认 benchmark/drafts/<prefix>.json）。
2. 读取该 thread 全部 chunks，基于真实存在的 section 名和内容起草题目：
   每个 thread 生成 10 题，配比 A1×5、A2×1、B×2、C×2。
3. A1/A2/C 题目的 expected_sections 必须是该论文里真实出现过的 section 名；expected_terms 必须是从对应块内容里挑出的、有区分度的英文专名/关键词（请在脚本里直接从 content 抽取，例如含大写的专名、关键术语，不要臆造）。
4. B 类题为通用领域知识（与该论文无关），expected_sections 与 expected_terms 留空。
5. 输出合法 JSON（ensure_ascii=False, indent=2），并在终端打印每题 id、类别、章节、关键词供人工核对；不要自动合并进 questions.json。

约束：
- 出题过程尽量基于规则读取 chunks，不调用大模型；如确需调用，先用 config.py 里的模型并打印将产生的调用次数。
- 不要修改 app.py、query_router.py、rag_store.py 的现有逻辑。
- 完成后运行 python -m unittest discover -s tests 确认无回归，并用一个本地已存在的 thread_id 试跑一次，把样例输出路径告诉我。
```

---

## 提示词 P2：批量生成答案并存档（供人工评阅）

```text
我需要让 Research Copilot 把评测题真实回答一遍并把结果存档，以便人工评阅引用和幻觉，而不用在网页逐题手问。请新增一个脚本，不要改动现有在线接口行为。

现状：
- 题库 benchmark/questions.json（每题含 id/question/expected_category/expected_sections/expected_terms/thread_id）。
- 路由与证据构建在 app.py：build_question_evidence(rag, thread_id, route)；query_router.build_route(question) 返回含 category 与 needs_retrieval 的 dict。
- Agent 在 agent_setup.py：create_research_agent() 用于脚本；模型见 config.py。
- B 类问题不检索。

请实现 benchmark/generate_answers.py：
1. 参数：--cases benchmark/questions.json、--output benchmark/answers/answers.jsonl、--limit（可选，便于先小批量试跑）、--ids（可选，只跑指定 id）。
2. 对每道题：构建路由→按需构建证据→调用 agent 生成最终答案（用与线上一致的证据/提示词路径）。
3. 每行写出一条 JSON：{id, thread_id, question, category, evidence, evidence_blocks, evidence_chars, answer, input_tokens, output_tokens, error}。
   token 从模型返回消息的 usage_metadata 取，取不到记 null。
4. 逐题落盘（跑一题写一题），中途失败不要丢掉前面结果；失败题写 error 后继续。
5. 终端打印进度和成功/失败计数。

约束：
- 这会真实调用模型产生费用：默认必须先要求 --limit 才能跑全量，或在脚本开头打印"将生成 N 个答案"并要求确认；先保证用 --limit 2 跑通。
- 答案与证据属于私有内容，输出放在 benchmark/answers/，并确认该目录已被 .gitignore 忽略；不要提交。
- 不要降低证据铁律或跳过路由。
- 完成后跑 python -m unittest discover -s tests，并用 --limit 2 试跑，把输出样例给我看。
```

---

## 提示词 P3：检索优化与前后对照（攻实体块落选）

```text
Research Copilot 存在一个已量化的检索短板：路由正确、章节也对，但包含关键实体的块没被选进证据。例如 benchmark/questions.json 中 s-a1-04 的 "Pile"（全文 10 次）、a-c-01 的 "Kakeya"（全文 19 次）会落选，因为问题里没出现这些专名，纯向量语义检索不够"近"。

请在不破坏现有默认行为的前提下，实现一个可对比的检索增强方案。

现状：
- 检索逻辑在 rag_store.py（PaperRAG.search，向量召回 + MMR；list_chunks 返回全部块）。
- 证据构建在 app.py 的 build_question_evidence。
- 评测脚本 benchmark/run_benchmark.py 输出 routing_accuracy、retrieval_hit_rate 与每题明细。

请按下面做：
1. 在 rag_store.py 增加一种"实体感知/混合"选块策略，用开关控制，默认关闭（保持线上现有行为）。可选实现（挑你认为最稳妥的一种，并说明理由）：
   a. 混合检索：向量召回之外，对问题或目标章节做关键词匹配（如对 chunks 做简单 BM25/词项重叠），合并候选后再 MMR；
   b. 实体加权重排：从候选块里抽取大写专名/术语，对包含稀有实体的块在重排时加权。
2. 让 benchmark/run_benchmark.py 支持参数 --retrieval-mode（baseline / experimental），其余流程不变，输出到不同 report 文件。
3. 用同一份 questions.json 分别跑 baseline 和 experimental，生成对比：总体与各类别（A1/A2/C）的 retrieval_hit_rate、每题命中变化（哪些题由 miss 变 hit、有没有反向退化）、retrieval_ms 与证据字符差异。
4. 为新选块逻辑补充离线单测（构造若干 fake chunks，验证含关键实体的块能被选中），放进 tests/。

约束：
- 不要引入需要联网下载模型的重型 reranker；优先纯 Python/现有依赖。
- 任何"提升"都以同一题库对照为准，不许改 questions.json 的 expected_* 来凑分。
- 完成后运行 python -m unittest discover -s tests，并把 baseline 与 experimental 的对比结论给我（含是否有退化），由我决定是否把 experimental 切成默认。
```

---

## 提示词 P4：多次报告聚合与趋势

```text
请为 Research Copilot 的评测写一个报告聚合脚本，方便我跟踪规模扩大和检索优化前后的变化。

现状：benchmark/reports/ 下有多个 run_benchmark 产出的 JSON（字段：generated_at, case_count, routing_accuracy, retrieval_hit_rate, results[]；每个 result 含 id, expected_category, actual_category, route_correct, retrieval_hit, expected_sections, actual_sections, evidence_chars, retrieval_ms 等）。

请实现 benchmark/aggregate_reports.py：
1. 输入：一个或多个 report 文件（或 --dir benchmark/reports）。
2. 输出对比表（同时打印并可写 --output JSON/Markdown）：
   - 每个 report 的 generated_at、case_count、routing_accuracy、retrieval_hit_rate；
   - 按 expected_category（A1/A2/B/C）拆分的路由准确率与检索命中率；
   - 按 thread_id（论文）拆分的检索命中率；
   - 相邻两次运行中，由 hit 变 miss、由 miss 变 hit 的题目 id 列表。
3. 纯读取，不修改任何 report 和 questions.json。

约束：报告文件可能缺字段或为 None，要稳健处理；完成后跑 python -m unittest discover -s tests，并用现有 benchmark/reports 下的文件实际运行一次给我看输出。
```

---

## 完成后的自检清单

- [ ] `python -m unittest discover -s tests` 全绿
- [ ] questions.json 是合法 JSON、id 无重复、类别配比符合预期
- [ ] 所有"提升"都有同一题库的前后对照，没有靠改标准答案凑分
- [ ] benchmark/answers/、.env、论文原文未被 git 跟踪
- [ ] 简历里的每个数字都能在 report/review.csv 里找到出处
