# Email Triage and Response (Streamlit App)

A Streamlit chat front end for an email triage and response agent. Users log in, paste the body of an email, and the app runs it through a LangGraph workflow that classifies the email, looks up the member's details in a SQL database, searches the policy knowledge base, and drafts a reply.

The UI lives in `app_streamlit.py`. The workflow it calls is built in `app/src/triage_graph.py`, which in turn uses two agents:

- `app/src/agent_member_info.py`: a SQL agent that looks up member details in a SQLite database.
- `app/src/agent_search_knowleadge.py`: a RAG agent over a Qdrant collection, with skills and memory.

## Features

- **Login gate**: username/password form backed by `st.secrets`, with constant-time password comparison (`hmac.compare_digest`).
- **Session handling**: logged-in user shown in the sidebar, with a log out button that clears the session.
- **Chat interface**: standard Streamlit chat UI with message history kept in `st.session_state`.
- **Per-session thread ID**: each session gets a thread ID of the form `<user>-<8 hex chars>`, passed to the graph (along with the user and the project, `kb`) so agent state can be tracked per user and session.
- **Structured triage**: a local LLM extracts the member number and decides whether the email is a policy/coverage question.
- **Member lookup**: a SQL agent queries a SQLite `member_info` table using the extracted member number.
- **Knowledge-base search**: a RAG agent retrieves relevant policy passages and answers using the email and the member's details.
- **Drafted reply**: the final node drafts a response from the email, the policy answer and the member info.
- **Agent guard rails**: both agents use model-call and tool-call limits, conversation summarisation, and a write gate on long-term memory that rejects secrets (passwords, tokens, etc.).
- **Error handling**: exceptions from the graph are shown in the chat as a friendly message rather than crashing the app.

## Triage workflow (`triage_graph.py`)

```
START -> triage_email -> [is policy question?]
                            |-- yes -> search_member_info -> search_knowledge_base -> draft_response -> END
                            |-- no  -> END
```

| Node | Purpose |
| --- | --- |
| `triage_email` | Uses a local LLM with structured output (`Triage`) to extract the member number and decide whether the email is a policy/coverage question |
| `search_member_info` | Runs `SearchMemberInfoAgent` (SQL agent over `member_info.db`) to retrieve the member's details |
| `search_knowledge_base` | Runs `SearchKnowledgeBaseAgent` (RAG over Qdrant, with skills and memory), passing it the email body and the member info |
| `draft_response` | Asks the LLM to draft a reply using the email, the knowledge-base answer and the member info |

### Graph state (`EmailState`)

| Field | Description |
| --- | --- |
| `email_body` | Raw email text entered in the chat |
| `user`, `thread`, `project` | Caller context passed to the agents (identity, session thread, project namespace) |
| `member_number` | Member number extracted by triage |
| `is_question_about_policy` | Triage decision used for routing |
| `member_info` | Final answer from the member info agent |
| `knowledge_base_results` | Final answer from the knowledge-base agent |
| `draft_response` | Drafted reply |

## Agents

Both agents are built with `langchain.agents.create_agent` on `qwen3:8b` via Ollama, and share the same structure:

- **Runtime context** (`Ctx`): `user_id` and `project_id`, supplied by the app and never by the LLM. Memory is namespaced by `(user_id, project_id, "memories")`.
- **Dynamic prompt** (`build_context`): rebuilds the system prompt on every model call from policies, skills and long-term memory.
- **Skills**: text procedures loaded from `skill.md` files (YAML front matter with `name` and `description`, then the body). Tools `load_skill` and `read_skill_resource` expose them to the model.
- **Memory**: `save_memory` and `search_memories` tools backed by an `InMemoryStore` (lost on restart), plus an `InMemorySaver` checkpointer for per-thread short-term memory.
- **Middleware**: `SummarizationMiddleware` (triggers at 3000 tokens, keeps the last 4 messages), `ModelCallLimitMiddleware` (10 per run) and `ToolCallLimitMiddleware` (6 per run).
- **`ask()` return value**: a dict with `user_query`, `final_answer`, `tools_called` (name, arguments, output) and `tokens_consumed`.

| | `SearchMemberInfoAgent` | `SearchKnowledgeBaseAgent` |
| --- | --- | --- |
| File | `agent_member_info.py` | `agent_search_knowleadge.py` |
| Data source | SQLite DB at `app/data/sql/member_info.db` | Qdrant collection `bupa cover` |
| Domain tools | LangChain `SQLDatabaseToolkit` tools | `search_knowledge_base` (top 5 passages, tagged `[document#chunk]`) |
| Skills loaded | `search-member-info` (always injected) | `knowledge-base-answering` (always injected); others via catalog + `load_skill` |

## Requirements

- Python 3.12+ (the graph uses f-strings that reuse the same quote type inside expressions)
- [Ollama](https://ollama.com) running locally at `http://localhost:11434` with the `qwen3:8b` model pulled
- A local Qdrant collection (named `bupa cover`) populated by `create_db.py` via `app.src.ragging`
- A SQLite database of members at `app/data/sql/member_info.db`, created by `create_member_info.py`
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- Python packages (installed by `uv sync`):
  - `streamlit`
  - `langchain`, `langgraph`, `langchain-ollama`, `langchain-core`, `langchain-community`
  - `sqlalchemy` (used by the SQL agent)
  - `pandas` (used to load the member CSV)
  - `pydantic`, `pyyaml`
  - Qdrant client and any dependencies of `app.src.ragging`

## Setup

1. **Install dependencies** with [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync
   ```

   This creates a `.venv` and installs the packages listed in `pyproject.toml` (pinned by `uv.lock`). If you are starting without a `pyproject.toml`, create one with the packages from [Requirements](#requirements):

   ```bash
   uv init
   uv add streamlit langchain langgraph langchain-ollama langchain-community sqlalchemy pandas pydantic pyyaml qdrant-client
   ```

2. **Pull the model**

   ```bash
   ollama pull qwen3:8b
   ```

3. **Configure users** in `.streamlit/secrets.toml`:

   ```toml
   [passwords]
   alice = "change-me"
   bob = "change-me-too"
   ```

   Keys are usernames and values are passwords. Do not commit this file.

4. **Create the member database** from `data/member_info.csv`:

   ```bash
   uv run python create_member_info.py
   ```

   The CSV is loaded into a `member_info` table (`member_number` as `INTEGER`, `product` as `TEXT`) in `app/data/sql/member_info.db`. Any existing table is replaced. Parent folders are created if missing.

5. **Index the policy document** into the Qdrant collection with `create_db.py`:

   ```bash
   uv run python create_db.py
   ```

   This reads `data/bupa cover.txt`, chunks it, and upserts the chunks into the `bupa cover` collection.

6. **Add the agent skills** (see [Project structure](#project-structure)). Skill files must be named `skill.md` (lowercase), as the loaders glob for that exact name.

## Running

From the project root, so the `app.src...` imports resolve:

```bash
uv run streamlit run app_streamlit.py
```

Then open the URL Streamlit prints (usually http://localhost:8501), log in, and paste an email into the chat box. The email should include the member's number, since triage extracts it.

## Project structure

```
.
├── app_streamlit.py                    # Streamlit UI
├── pyproject.toml                      # dependencies (used by uv sync)
├── uv.lock                             # locked versions
├── create_db.py                        # Chunks and indexes the policy text into Qdrant
├── create_member_info.py               # Loads member_info.csv into SQLite
├── .streamlit/
│   └── secrets.toml                    # user credentials (not committed)
├── data/
│   ├── bupa cover.txt                  # policy document to index
│   └── member_info.csv                 # member data to load
└── app/
    ├── data/
    │   └── sql/
    │       └── member_info.db          # SQLite database (generated)
    └── src/
        ├── triage_graph.py             # LangGraph email triage workflow
        ├── agent_member_info.py        # SQL agent for member lookup
        ├── agent_search_knowleadge.py  # RAG agent with skills + memory
        ├── ragging.py                  # Chunker and Qdrant collection wrapper
        └── skills/                     # resolved relative to the agent files
            ├── search-member-info/
            │   └── skill.md
            └── knowledge-base-answering/
                └── skill.md
```


## Security notes

- Passwords are stored in plain text in `st.secrets`. Fine for a prototype, but for production use hashed passwords or an SSO provider.
- Login attempts are not rate limited.
- The app is intended to run behind HTTPS if exposed beyond localhost.
- The SQL agent has access to every table in `member_info.db` through the standard SQL toolkit. For production, use a read-only database connection and restrict the tables it can see.
- Member details are passed through local models only (Ollama), so no data leaves the machine, but the agents' memory notes are stored unencrypted in process memory.