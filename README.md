# CSIT5900 SmartTutor – Multi-turn Homework Tutoring Agent

This repository contains the **current full SmartTutor project** for the HKUST CSIT5900 2025–26 Spring course project.

**Recommended entry point:** use the **web UI** in `webui.py`. It is the primary interface, exposes the most complete user experience, and is the best way to demonstrate the project. The CLI in `demo.py` is still included as a secondary terminal-based alternative.

---

## What this project includes

SmartTutor is a production-style homework tutoring agent with:

- **Multi-turn conversation memory** with token-aware history truncation
- **Homework-only guardrails** for in-scope subject filtering and refusal handling
- **Normal mode** and **Strict mode** response pipelines
- **Web UI with streaming replies** and visible strict-mode stage cards
- **Optional live search** with source review before evidence is shown or passed to generation
- **Academic level adaptation** based on user statements
- **Conversation summary** and **practice exercise generation**
- **Follow-up suggestion chips** in the web UI
- **Two interchangeable LLM backends**: Azure OpenAI and One API (OpenAI-compatible)
- **Automated unit tests** for guardrails, response handling, and search filtering

Supported subjects:

- **Mandatory:** Math, History
- **Optional:** Geography, Finance, Economics, Philosophy, Chemistry

---

## Recommended usage: Web UI first

> If you only use one interface, use **`webui.py`**.

### 1. Prepare the environment

If you use the repository's Conda environment:

```bash
conda activate agent-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create local environment variables:

```bash
cp .env.example .env
```

Fill in `.env` with either:

- **Azure OpenAI** credentials, or
- **One API** credentials

Minimum required configuration:

```dotenv
LLM_BACKEND=azure
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

Or:

```dotenv
LLM_BACKEND=oneapi
ONEAPI_API_KEY=your-key
ONEAPI_BASE_URL=https://your-endpoint/v1
```

### 2. Start the web UI

```bash
python webui.py
```

Then open:

- [http://localhost:8000](http://localhost:8000)

You can also use the shortcut:

```bash
./start.sh
```

`start.sh` simply runs `python webui.py`.

### 3. Why the web UI is the recommended way to use SmartTutor

The web UI exposes the richest version of the project:

- **Streaming assistant replies**
- **Normal / Strict mode switch**
- **Auto / On / Off search switch**
- **Allowed subjects selector** with Math and History always enabled
- **Starter prompt cards** for quick demo use
- **Visible strict-mode review pipeline** with stage-by-stage status
- **Sources drawer** for inspected search results
- **Follow-up suggestion chips** after each answer
- **Markdown + LaTeX rendering** for assistant responses
- **New chat reset** from the UI

### 4. How to use the web UI

1. Open the homepage.
2. Ask a homework question, or click a starter prompt.
3. Optionally switch:
   - **Mode**: `Normal` or `Strict`
   - **Search**: `Auto`, `On`, or `Off`
   - **Subjects**: enable or disable optional subjects
4. Read the streamed reply.
5. If search was used, open the **Sources** button to inspect retrieved summaries.
6. Click a follow-up chip to continue the conversation.
7. Use **New chat** to clear history.

### 5. Web API endpoints used by the UI

`webui.py` also exposes these endpoints:

- `GET /` – main HTML interface
- `POST /api/chat` – non-streaming chat response
- `POST /api/chat/stream` – streaming SSE chat response
- `POST /api/followups` – generate/regenerate follow-up suggestions
- `POST /api/clear` – clear conversation history
- `GET /api/demos` – starter prompts for the frontend

### 6. Notes for first-time startup

- If the frontend loads but the backend is not configured, the page still opens.
- Chat requests will return a **clear configuration error** until `.env` is completed.

---

## Secondary usage: CLI demo (`demo.py`)

Use the CLI mainly for:

- quick terminal-only demos
- manual regression checks
- comparing normal vs strict behavior without the browser

### Run the CLI

```bash
python demo.py
```

### CLI controls

| Command | Meaning |
|---|---|
| `demo-math` | Valid math question |
| `demo-history` | Valid history question |
| `demo-geography` | Valid geography question |
| `demo-reject1` | Non-homework travel request |
| `demo-reject2` | Local institutional trivia / off-scope history-like request |
| `demo-summary` | Conversation summary request |
| `demo-level` | Academic level statement |
| `demo-exercise` | Practice exercise generation |
| `subjects` | Review or change allowed subject scope |
| `mode normal` | Use normal mode |
| `mode strict` | Use strict mode |
| `search auto` | Search only when needed |
| `search on` | Always search |
| `search off` | Disable search |
| `status` | Show current mode/search state |
| `clear` | Clear conversation history |
| `exit` / `quit` | Leave the program |

The CLI and the web UI share the same core tutoring logic, subject-scope logic, search service, and guardrails.

---

## Project structure (current repository)

The tree below covers the maintained source files and support files in the current project. Generated caches, Git metadata, local IDE settings, and local secrets such as `.env` are intentionally not documented here as project source.

```text
.
├── .env.example                 # Environment-variable template
├── AGENTS.md                    # Repository-specific coding/contribution instructions
├── README.md                    # This file
├── agent-home-work-azure.py     # Older reference/prototype implementation
├── config/
│   ├── __init__.py
│   └── settings.py              # Central runtime settings, prompts, subjects, feature flags
├── core/
│   ├── __init__.py
│   ├── conversation.py          # Conversation memory and token-based truncation
│   ├── guardrails.py            # Local input pre-filter and academic-level detection
│   ├── response_handler.py      # Normal/strict orchestration, summaries, follow-ups
│   └── search.py                # Optional live search, source normalization, search review
├── demo.py                      # CLI demo entry point
├── guidance.md                  # Build/assignment guidance used during project development
├── llm/
│   ├── __init__.py              # LLM client factory
│   ├── azure_client.py          # Azure OpenAI async client with retry logic
│   ├── base_client.py           # Abstract LLM client interface
│   └── oneapi_client.py         # One API async client with retry logic
├── project.pdf                  # Course/project reference document
├── requirements.txt             # Python dependencies
├── start.sh                     # Web UI launcher shortcut
├── static/
│   ├── favicon.svg              # Browser tab icon
│   └── index.css                # Web UI styling
├── templates/
│   └── index.html               # Web UI HTML + client-side JavaScript
├── tests/
│   ├── test_guardrails.py       # Guardrail unit tests
│   ├── test_response_handler.py # Normal/strict pipeline tests
│   └── test_search.py           # Search reviewer/filtering tests
└── webui.py                     # FastAPI app, UI routes, chat API, streaming API
```

### Main entry points

- **`webui.py`** – recommended main interface
- **`demo.py`** – terminal-based alternative

### Supporting / legacy files

- **`agent-home-work-azure.py`** – an older prototype kept for reference; it is **not** the main maintained app path
- **`guidance.md`** – project-building guidance and original requirement prompt
- **`project.pdf`** – project/course reference material

> Older materials may mention `single_file_demo.py`, but that file is **not part of the current maintained repository**.

---

## Core architecture

### Runtime flow

```text
Web UI or CLI
    ↓
ConversationManager
    ↓
Local guardrails / prefilter
    ↓
Optional SearchService
    ↓
ResponseHandler
    ├── Normal mode: single answer pipeline
    └── Strict mode: reviewer → generator → auditor
    ↓
Assistant reply + optional sources + follow-up suggestions
```

### Main modules

#### `config/settings.py`
Centralizes:

- backend selection
- subject configuration
- search settings
- strict-mode settings
- system prompts
- demo prompt definitions
- follow-up suggestion prompt templates

#### `core/conversation.py`
Handles:

- stored message history
- token counting with `tiktoken`
- hard turn limits
- academic-level state
- conversation reset

#### `core/guardrails.py`
Implements the local pre-filter that can reject requests before the LLM call. It includes checks for:

- empty input
- base64 / rot13 / morse normalization
- jailbreak and prompt-exfiltration attempts
- disguised service requests
- clearly out-of-scope subjects
- local institutional / organizational trivia disguised as history
- harmful or cheating-style requests
- academic-level statements
- conversation-summary requests

#### `core/response_handler.py`
Coordinates the full response pipeline:

- normal mode request handling
- strict mode reviewer / generator / auditor flow
- local handling of academic-level and summary requests
- follow-up suggestion generation
- streaming support for the web UI

#### `core/search.py`
Provides optional live search with source normalization and filtering. The current implementation can combine:

- DuckDuckGo Instant Answer API
- Wikipedia
- OpenAlex
- arXiv
- PubMed
- configured knowledge pages

Before sources are used, the project can optionally:

- generate query variants with a dedicated query-optimizer model
- filter retrieved evidence with a dedicated search-reviewer model

#### `llm/`
Provides a backend abstraction so the same tutoring logic can run on:

- **Azure OpenAI**
- **One API** (OpenAI-compatible)

Both client implementations support async chat calls and streaming.

#### `templates/index.html` + `static/index.css`
Together they implement the complete frontend, including:

- streaming chat UI
- strict-mode stage cards
- subject selector
- source drawer
- follow-up chips
- starter prompt rotation
- Markdown/LaTeX rendering

---

## Configuration reference

### Required backend settings

Use one backend:

#### Azure OpenAI

```dotenv
LLM_BACKEND=azure
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-08-01-preview
```

#### One API

```dotenv
LLM_BACKEND=oneapi
ONEAPI_API_KEY=your-key
ONEAPI_BASE_URL=https://your-endpoint/v1
```

### Important model-name note

The **current default main-model names** are set directly in `config/settings.py`:

- Azure default deployment: `gpt-4o`
- One API default model: `DeepSeek-V3.2`

If you want to change the **main default role** model, edit `config/settings.py`.

Role-specific overrides such as `QUERY_OPTIMIZER_*`, `SEARCH_REVIEWER_*`, `FOLLOWUP_*`, and `STRICT_*` are read from environment variables.

### Optional search settings

Supported by the current codebase:

```dotenv
SEARCH_ENABLED=true
SEARCH_KNOWLEDGE_PAGES_JSON=[{"name":"Course Notes","url":"https://example.com/notes","keywords":["calculus","limits"]}]
QUERY_OPTIMIZER_AZURE_DEPLOYMENT_NAME=gpt-4o
QUERY_OPTIMIZER_ONEAPI_MODEL_NAME=gpt-4o
QUERY_OPTIMIZER_TEMPERATURE=0.1
QUERY_OPTIMIZER_MAX_TOKENS=512
QUERY_OPTIMIZER_TIMEOUT_SECONDS=20
QUERY_OPTIMIZER_QUERY_COUNT=10
SEARCH_REVIEWER_AZURE_DEPLOYMENT_NAME=gpt-4o
SEARCH_REVIEWER_ONEAPI_MODEL_NAME=gpt-4o
SEARCH_REVIEWER_TEMPERATURE=0.0
SEARCH_REVIEWER_MAX_TOKENS=512
SEARCH_REVIEWER_TIMEOUT_SECONDS=20
SEARCH_MAX_MERGED_SOURCES=12
SEARCH_MAX_SOURCES_PER_QUERY=6
```

### Optional strict-mode settings

Supported by the current codebase:

```dotenv
STRICT_MODE_ENABLED=true
STRICT_REVIEWER_AZURE_DEPLOYMENT_NAME=gpt-4o
STRICT_REVIEWER_ONEAPI_MODEL_NAME=gpt-4o
STRICT_REVIEWER_TIMEOUT_SECONDS=30
STRICT_GENERATOR_AZURE_DEPLOYMENT_NAME=gpt-4o
STRICT_GENERATOR_ONEAPI_MODEL_NAME=gpt-4o
STRICT_GENERATOR_TIMEOUT_SECONDS=60
STRICT_AUDITOR_AZURE_DEPLOYMENT_NAME=gpt-4o
STRICT_AUDITOR_ONEAPI_MODEL_NAME=gpt-4o
STRICT_AUDITOR_TIMEOUT_SECONDS=30
STRICT_MAX_GENERATION_ATTEMPTS=3
```

### Optional follow-up suggestion settings

```dotenv
FOLLOWUP_AZURE_DEPLOYMENT_NAME=gpt-4o
FOLLOWUP_ONEAPI_MODEL_NAME=gpt-4o
FOLLOWUP_TEMPERATURE=0.3
FOLLOWUP_MAX_TOKENS=256
FOLLOWUP_SUGGESTER_TIMEOUT_SECONDS=15
```

---

## Testing and verification

The repository now includes automated tests under `tests/`.

### Run unit tests

```bash
python -m unittest discover -s tests -v
```

### Quick syntax check

```bash
python -m compileall .
```

### Manual checks

At minimum, verify that both entry points start cleanly:

```bash
python webui.py
python demo.py
```

### What the tests currently cover

- local guardrail decisions
- encoded / obfuscated unsafe-input rejection
- normal-mode refusal-before-generation behavior
- strict-mode fail-closed behavior
- multi-turn summary handling
- follow-up practice request handling
- search reviewer filtering before UI/generation exposure

---

## Built-in demo prompts and expected scenarios

The project includes built-in prompts for the core course-report scenarios:

1. **Valid Math** – `Is square root of 1000 a rational number?`
2. **Valid History** – `Who was the first president of France?`
3. **Invalid Non-homework** – `I need to travel to London from Hong Kong. What is the best way?`
4. **Invalid Off-scope History-like Query** – `Who was the first president of Hong Kong University of Science and Technology in Hong Kong?`
5. **Conversation Summary** – `Can you summarise our conversation so far?`
6. **Academic Level** – `I'm a university year one student, provide your answers accordingly.`
7. **Practice Exercises** – `I want to practice calculus for my final in math101, can you give me a few exercises?`
8. **Geography Support** – `What causes monsoon climates?`

---

## Notes for contributors

- Keep secrets in `.env`, never in committed source files.
- Update both `.env.example` and the README when new configuration is added.
- Keep new modules close to the feature they extend.
- Do not edit generated `__pycache__` files.

---

## License / usage

This project is maintained for **academic use** in the HKUST CSIT5900 2025–26 Spring course context.
