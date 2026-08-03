# AI Research Copilot — 产品需求与系统设计

## 一、产品定位

一句话：**一个帮助研究生高效阅读、理解和分析学术论文的 AI 科研助手。**

目标用户：研0/研一学生、需要大量读论文的科研人员。

核心假设：研究人员不缺获取论文的渠道，缺的是**理解论文的效率**和**跨论文的知识关联**。

---

## 二、用户故事（V1）

| ID | 用户故事 | 验收标准 |
|----|---------|---------|
| US-1 | 上传一篇 PDF 论文，系统自动生成结构化摘要（研究问题 / 方法 / 实验 / 结论） | 上传后 30s 内看到摘要，结构完整 |
| US-2 | 基于论文内容追问细节，如"为什么选这个方法？对比基线是什么？" | 回答引用论文具体章节，不凭空编造 |
| US-3 | 关闭页面后回来，还能看到之前的论文和对话记录 | 会话列表保存，点击可恢复 |

---

## 三、V1 功能列表

| 功能 | 优先级 | 说明 |
|------|--------|------|
| PDF 上传 | P0 | 支持上传学术 PDF，存到 OSS |
| PDF 解析 | P0 | PyMuPDF 提取文本，按章节结构化 |
| 结构化摘要 | P0 | LLM 自动生成论文结构化分析 |
| 论文问答 | P0 | 基于论文内容的对话式问答（RAG） |
| 会话记忆 | P0 | SqliteSaver + sessions 表（复用 ChefMate） |
| 流式输出 | P0 | SSE 逐字推送（复用 ChefMate） |

---

## 四、系统架构（V1）

```
用户操作                     后端                         数据/服务
┌──────────┐   PDF上传    ┌──────────────┐   Embedding   ┌─────────┐
│  前端     │  ─────────→ │ FastAPI       │ ───────────→ │ChromaDB │
│ index.html│             │              │               └─────────┘
│          │  ←── SSE ─── │ LangGraph     │   LLM 调用   ┌─────────┐
│ Tailwind │              │ Agent + Tools │ ───────────→ │ 通义千问 │
│ marked.js│              │ SqliteSaver   │              └─────────┘
└──────────┘              │ OSS 下载      │
                           │ PyMuPDF 解析  │
                           └──────────────┘
```

### Agent 工具箱（V1）

Agent 有 3 个工具可用：

1. **`retrieve_paper_content`** — RAG 检索工具，从 ChromaDB 查论文相关内容
2. **`get_paper_structure`** — 获取论文章节结构
3. **`search_web`** — Tavily 搜索（可选，用于搜索论文学术背景）

---

## 五、PDF 解析策略

```
PyMuPDF 提取原始文本
    │
    ▼
按学术论文关键词分段：
  Abstract | Introduction | Related Work |
  Method | Experiment | Result | Conclusion
    │
    ▼
每段为一个 chunk，保留 metadata：
  { paper_title, section, page_num, chunk_index }
    │
    ▼
Embedding → 存入 ChromaDB
```

关键决策：**不跨章节切片**。整个 Abstract 作为一个 chunk，整个 Method 作为一个 chunk。超出 512 token 的章节按段落拆。

---

## 六、V1 实施步骤

### Step 1：项目骨架（今天就做）
- 创建 `Research_Copilot/` 目录
- 从 ChefMate 复制：`config.py`、`database.py`、`schema.py`、`oss_sts.py`
- 新增：`prompts.py`（论文分析 prompt）
- 新增：`paper_parser.py`（PDF 解析 + 章节提取）
- 新增：`rag_store.py`（ChromaDB 封装）
- 改造：`agent_setup.py`（改名为 create_research_agent）
- 改造：`app.py`（/chat 端点适配 PDF 流程）
- 改造：`static/index.html`（PDF 上传 + 论文聊天 UI）

### Step 2：PDF 解析 + RAG
- 实现 paper_parser.py（PyMuPDF 文本提取 + 章节分割）
- 实现 rag_store.py（ChromaDB 初始化 + 存储 + 检索）
- 测试：上传一篇论文 → 解析成功 → 存入向量库

### Step 3：Agent + 前端联调
- 写论文分析 prompt
- Agent 调用 RAG 工具回答论文问题
- 前端展示结构化摘要和对话

---

## 七、ChefMate 复用清单

| 模块 | 复用方式 | 改动量 |
|------|---------|-------|
| `config.py` | 直接复制，去掉 Tavily | 删 3 行 |
| `database.py` | 复制，改 DB 路径 | 改 1 行 |
| `schema.py` | 直接复制 | 0 行 |
| `oss_sts.py` | 复制，改 session 名 | 改 1 行 |
| `app.py` 框架 | 复制后改造 /chat | 中等改动 |
| `agent_setup.py` | 复制后重命名 + 换工具 | 小改动 |
| `static/index.html` | 改造 UI | 大改动 |
| OSS 前端上传 | 复用 uploadToOSS | OSS 路径改 papers/ |

---

## 八、不做的事（V1 明确不做的）

- ❌ 多论文对比
- ❌ 论文库管理（上传多篇）
- ❌ 文献综述生成
- ❌ 公式/图表识别
- ❌ 论文推荐

这些放 V2/V3。
