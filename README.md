# Research Assistant Agent

> V1.0 — a single-paper, evidence-grounded research reading assistant.

Upload a PDF paper and receive a structured summary or follow-up answer whose paper facts are tied to retrieved source sections, PDF pages, and original excerpts. V1 is designed to make paper claims inspectable; it is not intended to replace reading a paper in full.

## What V1 does

- Upload one PDF per conversation through Alibaba Cloud OSS and temporary STS credentials.
- Parse PDF text with PyMuPDF, remove repeated header/footer noise, detect common sections, and retain page ranges per chunk.
- Generate a structured summary from bounded evidence.
- Answer follow-up questions with paper-grounded evidence and explicit source labels.
- Route questions into paper facts, paper limitations, general knowledge, or research reasoning.
- Persist conversations locally with SQLite and vectors with ChromaDB.
- Replace both vectors and conversation history when a new PDF is uploaded to the same conversation.

## Evidence and token controls

V1 does not register a retrieval tool with the model. Retrieval happens in Python before generation and evidence is injected only into the current request prompt.

- Evidence never becomes a `ToolMessage` and is not persisted in checkpoint history.
- A response is capped at 8 evidence blocks and 6000 characters of evidence.
- Up to 6 blocks come from semantic retrieval; remaining capacity is used for section coverage.
- Structural fallback reuses the same retrieval candidates and prefers semantically relevant chunks before page order.
- Reference and appendix chunks are excluded from factual evidence.
- General-knowledge questions bypass paper retrieval.

This keeps multi-turn token growth bounded: only normal user and assistant messages are retained, while paper excerpts are retrieved again when needed.

## Architecture

```text
Browser -> OSS upload -> FastAPI /chat
                         |
                         +-> PDF parser -> section chunks -> ChromaDB
                         |
                         +-> query router -> bounded evidence -> LangGraph agent
                                                          |
                                                          +-> SSE response

SQLite stores sessions and checkpoints. ChromaDB stores one paper collection per thread.
```

## Quick start

### 1. Create an environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your credentials. Do not commit `.env`.

```bash
copy .env.example .env
```

Required variables:

| Variable | Purpose |
|---|---|
| `DASHSCOPE_API_KEY` | DashScope-compatible chat and embedding API key |
| `DASHSCOPE_BASE_URL` | API base URL used by chat and embedding requests |
| `OSS_ACCESS_KEY_ID` / `OSS_ACCESS_KEY_SECRET` | Server-side credentials used to obtain STS tokens |
| `OSS_REGION` / `OSS_ROLE_ARN` | Alibaba Cloud STS configuration |
| `OSS_BUCKET` / `OSS_ENDPOINT` | Browser upload target |

### 3. Run

```bash
python -m uvicorn app:app --reload --port 8000
```

Open `http://127.0.0.1:8000`.

## Verification and maintenance

Run the white-box verification script from the project root:

```bash
python verify_token_mechanisms.py
```

It reads one existing local Chroma collection and uses a temporary database for write-oriented tests. It verifies evidence budgets, collection replacement, and ToolMessage cleanup.

Useful inspection command:

```bash
python inspect_session.py
```

SQLite databases may retain free pages after deleting conversations. With the service stopped, back up the database and run:

```sql
PRAGMA wal_checkpoint(TRUNCATE);
VACUUM;
PRAGMA integrity_check;
```

## V1 boundaries

- One paper per conversation; uploading another PDF resets that conversation's paper data and messages.
- PDF page labels refer to physical PDF pages, not printed page numbers.
- Section detection targets common English academic headings and cannot guarantee perfect extraction for every PDF layout.
- Retrieval uses vector similarity plus MMR, not a cross-encoder reranker.
- SQLite is appropriate for the single-user/local V1 workflow, not a high-concurrency deployment.
- A full-context model may be preferable for one-off, deep reading of a short paper; V1 prioritizes bounded, source-addressable multi-turn use.

## Project layout

```text
app.py                       FastAPI endpoints, SSE, evidence construction
agent_setup.py               LangGraph agent and SQLite checkpointer
paper_parser.py              PDF extraction, cleanup, section and page tracking
rag_store.py                 ChromaDB storage, embeddings, retrieval, MMR
query_router.py              A1/A2/B/C question routing
prompts.py                   Evidence-grounding prompt rules
database.py                  Session metadata and SQLite connection setup
static/index.html            Browser interface
verify_token_mechanisms.py   White-box verification script
inspect_session.py           Read-only conversation/vector inspection
Token优化复盘.md             Token-control design notes
INTERVIEW.md                 Interview-oriented project explanation
```

## Future direction

V1 is intentionally closed after single-paper evidence-grounded reading. A useful next project should not merely add more agents; it should solve a workflow that generic chat cannot reliably own, such as paper-to-code-to-experiment reproducibility and comparison for time-series research.
