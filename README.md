# Email Triage and Response (Streamlit App)

A Streamlit chat front end for an email triage and response agent. Users log in, paste the body of an email, and the app runs it through a LangGraph workflow that classifies the email, looks up policy information, and drafts a reply.

The UI lives in `app_streamlit.py`. The workflow it calls is built in `app/src/triage_graph.py`, which in turn uses the knowledge-base agent in `app/src/agent_search_knowleadge.py`.

## Features

- **Login gate**: username/password form backed by `st.secrets`, with constant-time password comparison (`hmac.compare_digest`).
- **Session handling**: logged-in user shown in the sidebar, with a log out button that clears the session.
- **Chat interface**: standard Streamlit chat UI with message history kept in `st.session_state`.
- **Per-session thread ID**: each session gets a thread ID of the form `<user>-<8 hex chars>`, passed to the graph so conversation state can be tracked per user and session.
- **Error handling**: exceptions from the graph are shown in the chat as a friendly message rather than crashing the app.


### Triage workflow (`triage_graph.py`)

```
START -> triage_email -> [is policy question?]
                            |-- yes -> search_knowledge_base ┐
                            |          search_member_info    ├-> draft_response -> END
                            |-- no  -> END
```

| Node | Purpose |
| --- | --- |
| `triage_email` | Uses a local LLM with structured output to extract the member number and decide whether the email is a policy/coverage question |
| `search_knowledge_base` | Runs the knowledge-base agent (RAG over a Qdrant collection, with skills and memory) |
| `search_member_info` | Placeholder for member lookup |
| `draft_response` | Placeholder for the final reply |

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally at `http://localhost:11434` with the `qwen3:8b` model pulled
- A local Qdrant collection (named `bupa cover`) populated by `app.src.ragging.Qdrantcollection`
- Python packages:
  - `streamlit`
  - `langchain`, `langgraph`, `langchain-ollama`, `langchain-core`
  - `pydantic`, `pyyaml`
  - Qdrant client and any dependencies of `app.src.ragging`

## Setup

1. **Install dependencies**

   ```bash
   pip install streamlit langchain langgraph langchain-ollama pydantic pyyaml qdrant-client
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


5. **Index your documents** into the Qdrant collection before running the app with `create_db.py`.

## Running

From the project root, so the `app.src...` imports resolve:

```bash
streamlit run app_streamlit.py
```

Then open the URL Streamlit prints (usually http://localhost:8501), log in, and paste an email into the chat box.

## Project structure

```
.
├── app_streamlit.py                    # Streamlit UI (this file)
├── .streamlit/
│   └── secrets.toml                    # user credentials (not committed)
├── skills/
│   └── <skill-name>/SKILL.md           # agent skills
└── app/
    └── src/
        ├── triage_graph.py             # LangGraph email triage workflow
        ├── agent_search_knowleadge.py  # RAG agent with skills + memory
        └── ragging.py                  # Qdrant collection wrapper
```

## Security notes

- Passwords are stored in plain text in `st.secrets`. Fine for a prototype, but for production use hashed passwords or an SSO provider.
- Login attempts are not rate limited.
- The app is intended to run behind HTTPS if exposed beyond localhost.

