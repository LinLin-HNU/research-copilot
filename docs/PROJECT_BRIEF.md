# Research Copilot — 项目说明（PROJECT BRIEF）

> 面向 **Claude / Codex 等 AI 助手**：读完这份就能快速明白项目做什么、怎么跑、文件怎么分工、哪些能改哪些是边界。
> 开发者复盘（历史难题、真实现状、评测/简历行动）见 `docs/DEV_JOURNAL.md`。本文件第 9 节附了可直接使用的 Codex 辅助提示词。

---

## 1. 项目定位

单篇论文的**证据绑定（evidence-grounded）阅读助手**。用户上传一篇 PDF，系统解析章节/页码、向量检索，在**生成之前**把有预算的论文证据注入提示词，输出带【来源章节 + PDF 页码 + 原文引用】的结构化摘要和多轮问答。

要解决的问题：普通 LLM 会混淆「论文内容 / 模型训练知识 / 论文之后的技术发展」，产生幻觉和"历史倒置"。本项目让三者边界清晰、每个论文事实可溯源。

当前版本：**V1.1**（单篇、本地、可信优先）。

---

## 2. 技术栈

| 层 | 选型 |
|---|---|
| Web | FastAPI、StreamingResponse（SSE）、uvicorn；前端原生 `static/index.html` |
| Agent | LangGraph `create_agent`、`@dynamic_prompt` 运行时上下文、SqliteSaver 检查点 |
| 消息 | LangChain HumanMessage / AIMessage；模型真实 `usage_metadata` |
| 解析 | PyMuPDF（fitz）：去页眉页脚、章节识别、章节内切块、页码追踪 |
| 向量库 | ChromaDB PersistentClient，每会话集合 `paper_<thread_id>`，默认 L2/HNSW |
| 模型 | DashScope 兼容接口，qwen omni-flash（`enable_thinking=False`）；embedding 按 20 条分批 |
| 存储 | SQLite：会话元数据 + 检查点 + `request_metrics` 无内容遥测 |
| 上传 | 阿里云 OSS + STS 临时凭证，浏览器前端直传 |
| 交付 | Dockerfile / compose.yaml / GitHub Actions（**Docker 已配置尚未实操**） |

---

## 3. 架构与数据流

```text
浏览器
  │  1) /oss/token 拿 STS 临时凭证
  │  2) 前端直传 PDF 到 OSS
  ▼
FastAPI /chat  ──(异步)──►  从 OSS 下载 PDF
  │
  ├──► paper_parser：去噪声 → 章节识别 → 切块(带 section/page)
  │                              │
  │                              ▼
  │                         ChromaDB（每会话一个集合）
  │
  ├──► query_router：A1 / A2 / B / C 分类
  │
  ├──► rag_store.search / list_chunks → 有预算证据（B 类跳过检索）
  │
  ├──► LangGraph agent（证据经 dynamic_prompt 只注入当前请求）
  │
  └──► SSE 流式返回 + 写 request_metrics（不含正文内容）

SQLite：sessions / checkpoints / request_metrics
```

两类入口：
- **上传新论文后的自动总结**：固定 4 个英文摘要查询 + 结构性兜底，覆盖 abstract/method/results 等。
- **后续追问**：先路由，再按需检索证据；B 类不检索。
- **覆盖语义**：同一会话再传 PDF，新论文解析成功后**同时**清空该 thread 检查点历史并删除旧向量集合。

---

## 4. 关键设计约束（改动前必读）

1. **模型不注册检索工具**。检索在 Python 侧完成，证据通过 `@dynamic_prompt` 注入，**不产生 ToolMessage、不写进 checkpoint**。不要退回"Agent 自主循环调检索工具"。
2. **证据预算**：单块 ≤ 600 字符；每次 ≤ 8 块、总计 ≤ 6000 字符；≤ 6 块语义检索，其余章节覆盖；排除 references/appendix 与过短块。常量在 `app.py`。
3. **证据铁律 / 三类分层**在 `prompts.py`：检索文本是唯一事实来源，一级事实必须引用原文，论文未写要明说，B/C 不得伪装成论文原文。
4. **SQLite 跨线程**：checkpoint 由 LangGraph 后台线程写入，连接用 `check_same_thread=False` + WAL + busy_timeout（SqliteSaver 内部有锁）。不要改回强制同线程。
5. **模型必须关 thinking**：`extra_body={"enable_thinking": False}`，否则长 prompt 下会空流。
6. **遥测不含内容**：`request_metrics` 不存问题/证据/答案；改字段时保持这条隐私边界（有测试守护）。

---

## 5. 文件地图

```text
app.py                FastAPI 路由、SSE、证据构建、遥测记录
agent_setup.py        LangGraph agent、SqliteSaver 单例、动态提示词、清历史
paper_parser.py       PDF 提取、去噪声、章节识别、切块与页码
rag_store.py          ChromaDB 存储/检索/MMR/list_chunks/删除集合
query_router.py       A1/A2/B/C 路由（廉价 LLM，失败回退 A1）
prompts.py            基础系统提示词、证据铁律、三类分层
config.py             模型初始化（flash、关 thinking、超时、重试）
database.py           SQLite 连接、sessions、request_metrics
oss_sts.py            STS 临时凭证
schemas.py            请求/响应模型
metrics_report.py     遥测只读汇总（成功率、p50/p95、token）
inspect_session.py    会话/向量只读查看器
static/index.html     浏览器界面
benchmark/            评测：questions.json、run_benchmark.py、人工评阅模板与汇总
tests/test_core.py    离线单测（噪声页码、兜底排序、遥测隐私）
Dockerfile/compose.yaml/.github/   容器与 CI（Docker 未实操）
docs/                 本说明、复盘、评测行动；archive 存历史文档
```

---

## 6. 怎么跑

```bash
# 环境
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        # 填入 DASHSCOPE / OSS 凭证

# 启动（在 Research_Copilot 目录下，DB 用相对路径 resources/...）
python -m uvicorn app:app --reload --port 8000
# 打开 http://127.0.0.1:8000
```

测试与检查：
```bash
python -m unittest discover -s tests          # 离线单测
python verify_token_mechanisms.py             # 白盒机制验证
python inspect_session.py                     # 查看会话/向量
```

评测：
```bash
python benchmark/run_benchmark.py --cases benchmark/questions.json --output benchmark/reports/v2.json
python benchmark/summarize_human_review.py benchmark/review.csv
python metrics_report.py
```

数据位置：`resources/research_copilot.db`（SQLite）、`resources/` 下 ChromaDB 持久化目录。`.env`、论文原文、答案、真实 thread_id 不要提交。

---

## 7. 已知边界（V1.1）

- 一会话一篇论文；再传即重置该会话。
- 章节识别针对常见英文学术标题，不能保证适配所有排版。
- 检索为向量 + MMR，**无 cross-encoder 精排**；存在"章节对、关键实体块落选"的真实短板。
- Chroma 默认 L2 与 MMR 内部余弦度量不一致。
- 页码是 PDF 物理页码。
- SQLite 适合单用户本地，不适合高并发。
- `/oss/token` 无鉴权，**不可直接公网暴露**。

---

## 8. 路线图

### ⏳ V2 —— 【未来规划，尚未实现，勿当成已完成功能】
- 多论文知识库：跨会话持久化、统一检索。
- 跨论文对比分析（方法/数据集/指标）。
- 检索增强：混合检索（向量 + 关键词）或 cross-encoder 精排，先解决实体块落选问题。
- 可能用 LangGraph 编排 Planner → Retriever → Generator → Validator。

### ⏳ V3 —— 【未来规划，尚未实现】
- Research Agent：按主题自动检索相关论文。
- 自动生成文献综述（发展脉络 / 关键方法 / 代表论文 / 未来趋势）。
- 论文→代码→实验的复现与比较闭环（候选方向，尤其适合时间序列研究）。

> 注：V2/V3 是方向，不是承诺；动手前先用第 9 节提示词产出的评测体系验证收益，避免为堆功能而堆功能。

---

## 9. 附录：Codex 辅助提示词（P1–P4，可直接复制）

> 一次只贴一个，做完并自测后再进行下一个；要求 Codex 改完跑 `python -m unittest discover -s tests`，不要破坏现有测试。

### P1：自动出题草稿生成器

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
2. 读取该 thread 全部 chunks，基于真实 section 名和内容起草：
   每 thread 生成 10 题，配比 A1×5、A2×1、B×2、C×2。
3. A1/A2/C 的 expected_sections 必须是该论文真实出现过的 section 名；expected_terms 必须从对应块 content 里抽取有区分度的英文专名/关键词，不要臆造。
4. B 类题为通用领域知识（与该论文无关），expected_sections 与 expected_terms 留空。
5. 输出合法 JSON（ensure_ascii=False, indent=2），终端打印每题 id、类别、章节、关键词供人工核对；不要自动合并进 questions.json。

约束：出题尽量基于规则读 chunks、不调大模型；如确需调用先打印调用次数。不要改 app.py/query_router.py/rag_store.py 现有逻辑。完成后跑 python -m unittest discover -s tests，并用一个本地已存在的 thread_id 试跑，把输出路径告诉我。
```

### P2：批量生成答案并存档（供人工评阅）

```text
我需要让 Research Copilot 把评测题真实回答一遍并存档，以便人工评阅引用和幻觉，而不用在网页逐题手问。请新增脚本，不要改动现有在线接口行为。

现状：题库 benchmark/questions.json；app.py 有 build_question_evidence(rag, thread_id, route)；query_router.build_route(question) 返回含 category 与 needs_retrieval；agent_setup.py 的 create_research_agent() 用于脚本，模型见 config.py；B 类不检索。

请实现 benchmark/generate_answers.py：
1. 参数：--cases、--output benchmark/answers/answers.jsonl、--limit（可选）、--ids（可选）。
2. 对每题：路由 → 按需构建证据 → 调用 agent 生成最终答案（与线上证据/提示词路径一致）。
3. 每行写：{id, thread_id, question, category, evidence, evidence_blocks, evidence_chars, answer, input_tokens, output_tokens, error}；token 从 usage_metadata 取，取不到 null。
4. 跑一题写一题，失败题写 error 后继续，不丢前面结果；终端打印进度与成败计数。

约束：会真实产生费用——脚本开头打印"将生成 N 个答案"，要求 --limit 或确认才能跑全量，先保证 --limit 2 跑通。输出在 benchmark/answers/（应已被 .gitignore 忽略），不要提交。不要降低证据铁律或跳过路由。完成后跑 python -m unittest discover -s tests，用 --limit 2 试跑并给我看样例。
```

### P3：检索优化与前后对照（攻实体块落选）

```text
Research Copilot 有已量化的检索短板：路由正确、章节也对，但含关键实体的块没进证据。如 s-a1-04 的 "Pile"（全文 10 次）、a-c-01 的 "Kakeya"（19 次）会落选，因为问题里没出现这些专名，纯向量语义不够"近"。请在不破坏默认行为前提下，实现可对比的检索增强。

现状：检索在 rag_store.py（search：向量召回 + MMR；list_chunks：全部块）；证据构建在 app.py 的 build_question_evidence；评测 benchmark/run_benchmark.py 输出 routing_accuracy、retrieval_hit_rate 与每题明细。

请做：
1. 在 rag_store.py 增加"实体感知/混合"选块策略，开关控制、默认关闭。二选一（说明理由）：
   a. 混合检索：向量召回之外对目标章节做关键词匹配（简单 BM25/词项重叠），合并候选后再 MMR；
   b. 实体加权重排：从候选块抽取大写专名/术语，对含稀有实体的块加权。
2. 让 run_benchmark.py 支持 --retrieval-mode（baseline/experimental），输出到不同 report。
3. 用同一份 questions.json 分别跑，对比：总体与各类别 retrieval_hit_rate、每题 miss↔hit 变化（含反向退化）、retrieval_ms 与证据字符差异。
4. 为新逻辑补离线单测（fake chunks 验证含关键实体块能选中），放进 tests/。

约束：不引入需联网下载模型的重型 reranker；不许改 questions.json 的 expected_* 凑分。完成后跑 python -m unittest discover -s tests，把 baseline 与 experimental 对比结论（含是否退化）给我，由我决定是否切默认。
```

### P4：多次报告聚合与趋势

```text
请为 Research Copilot 写报告聚合脚本，跟踪规模扩大和优化前后的变化。benchmark/reports/ 下多个 report JSON，字段：generated_at, case_count, routing_accuracy, retrieval_hit_rate, results[]；result 含 id, expected_category, actual_category, route_correct, retrieval_hit, expected_sections, actual_sections, evidence_chars, retrieval_ms 等。

请实现 benchmark/aggregate_reports.py：
1. 输入一个或多个 report 文件（或 --dir benchmark/reports）。
2. 输出对比表（打印并可 --output JSON/Markdown）：
   - 每 report 的 generated_at、case_count、routing_accuracy、retrieval_hit_rate；
   - 按 expected_category 拆分的路由/检索命中率；
   - 按 thread_id 拆分的检索命中率；
   - 相邻两次运行中 hit↔miss 的题目 id 列表。
3. 纯读取，不改任何 report 和 questions.json。

约束：缺字段或 None 要稳健处理。完成后跑 python -m unittest discover -s tests，并用现有 reports 实际运行给我看输出。
```
#### dockers配置
- 目前已配置dockers镜像
##### 启动已有镜像
- 代码和依赖没有变化时：
- docker compose -f compose.local.yaml up -d
- -d 表示后台运行，终端可以继续使用。
- 修改代码后重新构建
>docker compose -f compose.local.yaml up -d --build
- 查看状态
>docker compose -f compose.local.yaml ps
- 查看日志
>docker compose -f compose.local.yaml logs -f --tail 100
- 其中：
- -f：持续显示新日志。
- --tail 100：先显示最近 100 行。
- 按 Ctrl+C 只会退出日志查看，不会停止容器。

- 停止但保留容器:
>docker compose -f compose.local.yaml stop
- 恢复：
>docker compose -f compose.local.yaml start
- 停止并删除容器、网络:
>docker compose -f compose.local.yaml down
- 这会删除容器和 Compose 网络，但不会删除：
- 项目源代码
- Docker 镜像
- resources 中的数据
- .env
- 下次运行 up -d 会重新创建容器。
- 避免随意运行：
```bash
docker compose down -v
docker system prune
```
- 修改代码后 Docker 怎样更新
- 正常更新流程是：
```bash
git pull
docker compose -f compose.local.yaml up -d --build
docker compose -f compose.local.yaml ps
docker compose -f compose.local.yaml logs --tail 100
```

原理是：
1. 拉取新代码。
2. 根据新代码重新构建镜像。
3. 创建新容器替换旧容器。
4. 继续挂载原来的 resources。
5. 用状态和日志确认启动成功。
如果只是修改 .env，通常不需要重新构建镜像，但需要重新创建容器：

- 如果只是修改 .env，通常不需要重新构建镜像，但需要重新创建容器：
>docker compose -f compose.local.yaml up -d --force-recreate
>
