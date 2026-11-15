# Configuration

All settings are loaded from a `.env` file in the project root. Copy `.env.example` to get started:

```bash
cp .env.example .env
```

---

## File Storage

Controls where uploaded documents are stored.

| Variable | Default | Description |
| --- | --- | --- |
| `FILE_STORAGE_BACKEND` | `local` | `local` for the `data/` folder, `s3` for AWS S3 |
| `DATA_DIR` | `data` | Local directory for uploaded files |
| `AWS_S3_BUCKET` | — | S3 bucket name (required when backend is `s3`) |
| `AWS_S3_PREFIX` | — | Optional key prefix inside the bucket |
| `AWS_REGION` | — | AWS region for S3 operations |
| `AWS_S3_ENDPOINT_URL` | — | Custom S3-compatible endpoint (e.g. MinIO) |
| `AWS_ACCESS_KEY_ID` | — | AWS access key (or use instance profile) |
| `AWS_SECRET_ACCESS_KEY` | — | AWS secret key |
| `AWS_SESSION_TOKEN` | — | Optional session token for temporary credentials |
| `STORAGE_CACHE_DIR` | `.storage_cache` | Local cache directory when backend is `s3` |

> When using S3, the app syncs objects to a local cache for parsing. The cache is populated on startup and after uploads.

---

## LLM

Controls the language model used for answering questions.

Set `LLM_MODE=local` to use Ollama, or `LLM_MODE=hosted` for an OpenAI/Groq-compatible API.

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_MODE` | `local` | `local` (Ollama) or `hosted` (OpenAI/Groq) |
| `OLLAMA_LLM_MODEL` | `gemma4` | Ollama model name for local inference |
| `HOSTED_LLM_PROVIDER` | `openai` | `openai` or `groq` |
| `HOSTED_LLM_MODEL` | `gpt-4o-mini` | Model name for the hosted provider |
| `HOSTED_LLM_API_KEY` | — | API key for the hosted provider |
| `HOSTED_LLM_BASE_URL` | — | Custom base URL for OpenAI-compatible APIs |
| `HOSTED_LLM_TEMPERATURE` | `1` | Sampling temperature (0–2) |
| `HOSTED_LLM_TOP_P` | `1` | Top-p (nucleus) sampling |
| `HOSTED_LLM_MAX_TOKENS` | `2048` | Maximum tokens in the response |
| `GROQ_REASONING_EFFORT` | `medium` | Reasoning effort level for Groq models |

### Fallback API keys

These are checked when provider-specific keys are not set:

| Variable | Used by |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI LLM, embeddings, and chunking |
| `GROQ_API_KEY` | Groq LLM |

> **Legacy:** `LOCAL_LLM_HOSTED=true` is still recognized for backward compatibility but ignored when `LLM_MODE` is set.

---

## Embeddings

Controls how document text is converted to vectors. This is independent of `LLM_MODE`.

Set `EMBEDDING_MODE=local` for Ollama, `EMBEDDING_MODE=custom` for built-in hash embeddings, or `EMBEDDING_MODE=hosted` for a hosted provider.

| Variable | Default | Description |
| --- | --- | --- |
| `EMBEDDING_MODE` | `local` | `local`, `custom`/`hash`, or `hosted` |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `LOCAL_EMBEDDING_DIMENSIONS` | `384` | Vector size for hash embeddings |
| `HOSTED_EMBEDDING_PROVIDER` | `openai` | `openai` or `nvidia` |
| `HOSTED_EMBEDDING_MODEL` | `text-embedding-3-small` | Model name for hosted embeddings |
| `HOSTED_EMBEDDING_API_KEY` | — | API key (falls back to `OPENAI_API_KEY`) |
| `HOSTED_EMBEDDING_BASE_URL` | — | Custom endpoint for the embedding API |

### Embedding modes explained

| Mode | Use case |
| --- | --- |
| `local` | Best quality for local dev — requires Ollama running |
| `custom`/`hash` | Deterministic, no external calls — useful for testing |
| `hosted` | Production-grade embeddings via OpenAI or NVIDIA |

---

## Chunking

Controls how parsed documents are split into smaller pieces before embedding.

If `CHUNKING_MODE` is not set, it inherits from `LLM_MODE`. When set to `hosted`, the app uses an LLM to perform semantic chunking (splitting by meaning). When `local`, no LLM is used: Excel rows pass through unsplit, and PDF pages / slides longer than `LOCAL_CHUNK_SIZE` characters are split by a rule-based `RecursiveCharacterTextSplitter` (ids become `{source}:...:chunk:{n}`, every chunk carries `chunk_index`, `0` for an unsplit document).

| Variable | Default | Description |
| --- | --- | --- |
| `CHUNKING_MODE` | value of `LLM_MODE` | `local` or `hosted` |
| `LOCAL_CHUNK_SIZE` | `1500` | Local mode: max characters per PDF/PPT chunk before splitting; `0` disables splitting (pure passthrough). Chunk ids depend on it, so run `POST /ai/reindex` once after upgrading or changing it (startup skips reindexing a non-empty store) |
| `OLLAMA_CHUNKING_MODEL` | value of `OLLAMA_LLM_MODEL` | Reserved; local chunking currently calls no LLM |
| `HOSTED_CHUNKING_PROVIDER` | `openai` | `openai` or `groq` |
| `HOSTED_CHUNKING_MODEL` | `gpt-4o-mini` | Hosted model for semantic chunking |
| `HOSTED_CHUNKING_API_KEY` | — | API key (falls back to `OPENAI_API_KEY`) |
| `HOSTED_CHUNKING_BASE_URL` | — | Custom base URL |
| `HOSTED_CHUNKING_TEMPERATURE` | `1` | Sampling temperature |
| `HOSTED_CHUNKING_TOP_P` | `1` | Top-p sampling |
| `HOSTED_CHUNKING_MAX_TOKENS` | `2048` | Max tokens for chunking responses |

> **Fallback:** If the hosted chunking LLM is unreachable, a rule-based `RecursiveCharacterTextSplitter` is used automatically.

> **Legacy names:** `HOSTED_LLM_CHUNKING_PROVIDER`, `HOSTED_LLM_CHUNKING_MODEL`, etc. are still recognized as fallbacks.

---

## Retrieval & Chat History

Tuning parameters for the retrieval pipeline and chat session memory.

| Variable | Default | Description |
| --- | --- | --- |
| `RETRIEVER_K` | `5` | Documents retrieved per query when the question doesn't name a file (one search across all files) |
| `RETRIEVER_MAX_FILE_CHARS` | `12000` | When the question names file(s) whose combined text is at most this many characters, the whole file(s) are sent to the LLM (complete answers for small spreadsheets). Lower it for low token-rate tiers |
| `RETRIEVER_K_PER_SOURCE` | `3` | Documents retrieved from each named file when the named files are too large to send whole |
| `MAX_CHAT_HISTORY_MESSAGES` | `12` | Messages retained per chat session |
| `MAX_CHAT_HISTORY_CONTENT_CHARS` | `2000` | Max characters kept per history message |

---

## Jira (via MCP)

The chat is **agentic**: the LLM decides which tools to call instead of always running a fixed retrieve-then-answer flow. It can search the knowledge base and, when Jira is enabled, query Jira (read-only) through a [Model Context Protocol](https://modelcontextprotocol.io/) server ([`mcp-atlassian`](https://github.com/sooperset/mcp-atlassian)).

| Variable | Default | Description |
| --- | --- | --- |
| `AGENT_ENABLED` | `true` | Master switch for agentic chat. When `false`, the classic RAG chain is used |
| `JIRA_ENABLED` | `false` | Enable the Jira MCP tools |
| `JIRA_URL` | — | Jira Cloud base URL, e.g. `https://your-site.atlassian.net` |
| `JIRA_USERNAME` | — | Atlassian account email |
| `JIRA_API_TOKEN` | — | Atlassian API token ([create one](https://id.atlassian.com/manage-profile/security/api-tokens)) |
| `JIRA_READ_ONLY` | `true` | Restrict the MCP server to read-only tools |
| `JIRA_ENABLED_TOOLS` | — | Optional comma-separated allow-list of MCP tools (empty = all read tools) |
| `JIRA_MCP_TRANSPORT` | `stdio` | `stdio` spawns the `mcp-atlassian` process; `streamable_http` connects to a running server |
| `JIRA_MCP_COMMAND` | `mcp-atlassian` | Console script to spawn for `stdio` transport |
| `JIRA_MCP_URL` | — | Server URL (only for `streamable_http` transport) |

> **Requires extra packages:** `pip install langchain-mcp-adapters mcp-atlassian`, plus the `JIRA_*` env vars above. These optional deps are pinned in `requirements.txt` (`langchain-mcp-adapters>=0.3,<0.4`, `mcp-atlassian>=0.23,<0.24`) because the networked MCP integration and the langgraph 1.x API move quickly; run `pip check` after installing to catch resolver conflicts.

> **Diagnosing load state:** `GET /ai/agent-status` reports whether the agent and its Jira tools actually loaded — `jira_loaded`, the bound `tools`, and a `jira_error` string when Jira degraded. Use it to tell a genuine failure apart from a silent fallback. If `JIRA_ENABLED=true` but the MCP packages are missing, `jira_error` names the exact `pip install` to run.

> **Tool count vs. token/rate limits:** enabling all Jira tools (~38) injects every tool's JSON schema into each agent request — several thousand extra tokens per call. On low-tier hosted LLM plans (e.g. Groq's 8000 tokens/minute) this can trip `413/429 rate_limit_exceeded`; the agent invocation then throws and falls back to the legacy chain, so the reply looks like "I don't have that information" even though Jira loaded fine. Restrict `JIRA_ENABLED_TOOLS` to the handful you actually need (a good starter set: `jira_search,jira_get_issue,jira_get_all_projects,jira_get_project_issues,jira_get_agile_boards,jira_get_sprints_from_board`) to keep requests small, or upgrade your provider tier.

> **Windows `stdio` PATH:** with `JIRA_MCP_TRANSPORT=stdio` the MCP client spawns `JIRA_MCP_COMMAND` (default `mcp-atlassian`) via `PATH`, so it must resolve to the installed console script. This works when uvicorn is launched from the activated venv. If you launch from outside the venv, set `JIRA_MCP_COMMAND` to the absolute path, e.g. `...\venv\Scripts\mcp-atlassian.exe`.

> **Needs a tool-calling LLM.** Hosted OpenAI/Groq support tools natively; for local Ollama use a tool-capable model (e.g. `llama3.1`, `qwen2.5`). The default `gemma4` **cannot** tool-call, so agentic features stay off and the app transparently falls back to the classic RAG chain. If the Jira connection or the agent build fails, the app logs a warning and serves a retrieval-only (or legacy) response — it never crashes on a missing MCP server.

---

## Vector Store

Controls where embeddings are persisted and searched.

Set `VECTOR_DB_BACKEND=chroma` (default) for ChromaDB, or `VECTOR_DB_BACKEND=hana` for SAP HANA Cloud Vector Engine.

### ChromaDB

| Variable | Default | Description |
| --- | --- | --- |
| `VECTOR_DB_BACKEND` | `chroma` | `chroma` or `hana` |
| `CHROMA_MODE` | `local` | `local` (persist to disk) or `http` (client-server) |
| `CHROMA_HOST` | `localhost` | Chroma server hostname (`http` mode only) |
| `CHROMA_PORT` | `8000` | Chroma server port (`http` mode only) |
| `CHROMA_SSL` | `false` | Use HTTPS for the Chroma connection |
| `CHROMA_HEADERS` | — | JSON string of auth headers, e.g. `{"Authorization": "Bearer token"}` |

### SAP HANA Cloud

| Variable | Default | Description |
| --- | --- | --- |
| `HANA_DB_ADDRESS` | — | HANA host (required when backend is `hana`) |
| `HANA_DB_PORT` | `443` | HANA port |
| `HANA_DB_USER` | — | Database username |
| `HANA_DB_PASSWORD` | — | Database password |
| `HANA_DB_TABLE_NAME` | `LANGCHAIN_VECTORS` | Table for vector storage |

> SAP HANA requires additional dependencies: `pip install hdbcli langchain-hana`

---

## Quick Reference

Minimal `.env` for local development with Ollama:

```env
LLM_MODE=local
OLLAMA_LLM_MODEL=gemma4
EMBEDDING_MODE=local
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

Minimal `.env` for hosted (OpenAI):

```env
LLM_MODE=hosted
HOSTED_LLM_PROVIDER=openai
HOSTED_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
EMBEDDING_MODE=hosted
HOSTED_EMBEDDING_PROVIDER=openai
```

Minimal `.env` for hosted (Groq + NVIDIA embeddings):

```env
LLM_MODE=hosted
HOSTED_LLM_PROVIDER=groq
HOSTED_LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_...
EMBEDDING_MODE=hosted
HOSTED_EMBEDDING_PROVIDER=nvidia
HOSTED_EMBEDDING_MODEL=NV-Embed-QA
HOSTED_EMBEDDING_API_KEY=nvapi-...
```
