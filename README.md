# CSIT5900 SmartTutor – Multi-turn Homework Tutoring Agent

A production-ready, multi-turn homework tutoring AI agent built for the HKUST CSIT5900 2025-26 Spring Semester Course Project. The agent uses LLM as its core engine with strict guardrails, conversation memory, and academic-level adaptation.

## Features

- **Multi-Subject Support** – Math and History homework tutoring (mandatory), with optional coverage for Geography, Finance, Economics, Philosophy, Chemistry.
- **Dual Guardrails** – Code-level pre-check (regex/heuristic) + LLM system-prompt enforcement to reject non-homework or off-subject inputs.
- **Multi-turn Conversation** – Full conversation memory with automatic token-based truncation to prevent context window overflow.
- **Academic Level Adaptation** – Adjusts answer depth based on the user's stated academic background.
- **Conversation Summary** – Generates summaries of the entire dialog history on request.
- **Practice Exercise Generation** – Creates targeted practice questions for specified subjects and levels.
- **Optional Live Search** – Adds free-source retrieval from DuckDuckGo, Wikipedia, OpenAlex, arXiv, PubMed, and configurable knowledge pages for fresher factual answers.
- **Dual API Backends** – Azure OpenAI API (project submission) and One API (OpenAI-compatible, self-testing), switchable via a single config toggle.
- **CLI Demo Interface** – Built-in demo shortcuts for the 1-minute demo recording.

## Project Structure

```
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── config/
│   ├── __init__.py
│   └── settings.py           # API configuration, guardrails rules, subject settings
├── core/
│   ├── __init__.py
│   ├── conversation.py       # Conversation memory & context management
│   ├── guardrails.py         # Input classification & guardrails logic
│   ├── response_handler.py   # Response generation & formatting
│   └── search.py             # Free-source retrieval and knowledge-page scraping
├── llm/
│   ├── __init__.py           # Factory function for LLM client selection
│   ├── base_client.py        # Abstract base class for LLM clients
│   ├── azure_client.py       # Azure OpenAI API implementation
│   └── oneapi_client.py      # One API (OpenAI-compatible) implementation
├── demo.py                   # Main CLI demo entry point (modular version)
├── webui.py                  # FastAPI web UI and chat API
├── start.sh                  # Convenience launcher for the web UI
├── templates/
│   └── index.html            # Frontend chat interface
├── single_file_demo.py       # Single-file combined version for quick demo
└── agent-home-work-azure.py  # Original reference implementation
```

## Prerequisites

- Python 3.10+
- An Azure OpenAI resource with a deployed model **OR** a One API endpoint

## Setup

### 1. Clone and install dependencies

```bash
cd agent-project
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

**For Azure OpenAI (project submission):**
```dotenv
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-08-01-preview
LLM_BACKEND=azure
```

**For One API (self-testing):**
```dotenv
ONEAPI_API_KEY=your-key
ONEAPI_BASE_URL=https://your-endpoint/v1
LLM_BACKEND=oneapi
```

Current code defaults in [config/settings.py](/Users/lijinchuan/Documents/HKUST/CSIT5900-AI/agent-project/config/settings.py):
- Azure deployment name: `gpt-4o`
- One API model name: `DeepSeek-V3.2`

If you want to change either default, update [config/settings.py](/Users/lijinchuan/Documents/HKUST/CSIT5900-AI/agent-project/config/settings.py) directly.

Optional search-related environment variables:

```dotenv
SEARCH_ENABLED=true
SEARCH_KNOWLEDGE_PAGES_JSON=[{"name":"Course Notes","url":"https://example.com/notes","keywords":["calculus","limits"]}]
```

`SEARCH_KNOWLEDGE_PAGES_JSON` should be a JSON array. Each item needs `name`, `url`, and `keywords`.

### 3. Run the demo

**Modular version** (recommended):
```bash
python demo.py
```

**Web UI**:
```bash
python webui.py
```

**Single-file version** (for quick demos):
```bash
python single_file_demo.py
```

You can also start the web UI with:
```bash
./start.sh
```

If the LLM backend is not configured yet, the web page can still load, but chat requests will return a clear configuration error until `.env` is completed.

## Web Search Modes

The web UI now includes three search modes above the message box:

- `Auto Search` – only searches when the query likely needs fresh facts or sources
- `Search On` – always retrieves web, academic, and matching knowledge-page sources
- `Search Off` – keeps responses local to the model and conversation history only

## Demo Shortcuts

Type any of these keywords at the prompt to trigger built-in test cases:

| Shortcut        | Test Case                                                                 |
|-----------------|---------------------------------------------------------------------------|
| `demo-math`     | Valid math question: *Is square root of 1000 a rational number?*           |
| `demo-history`  | Valid history question: *Who was the first president of France?*            |
| `demo-geography`| Valid geography question: *What causes monsoon climates?*                    |
| `demo-reject1`  | Non-homework rejection: *I need to travel to London from Hong Kong...*     |
| `demo-reject2`  | Off-subject rejection: *Who was the first president of HKUST?*             |
| `demo-summary`  | Conversation summary: *Can you summarise our conversation so far?*         |
| `demo-level`    | Academic level: *I'm a university year one student...*                     |
| `demo-exercise` | Practice exercises: *I want to practice calculus for my final in math101...* |
| `clear`         | Clear conversation history and start fresh                                 |
| `exit` / `quit` | Exit the program                                                           |

## Test Cases (for Project Report)

The agent has been tested against the following mandatory examples:

1. **Valid Math** – `Is square root of 1000 a rational number?` → Correct academic answer
2. **Valid History** – `Who was the first president of France?` → Concise historical answer
3. **Invalid (Non-homework)** – `I need to travel to London from Hong Kong...` → Rejected with reason
4. **Invalid (Off-subject)** – `Who was the first president of HKUST?` → Rejected with reason
5. **Conversation Summary** – `Can you summarise our conversation so far?` → Complete summary
6. **Academic Level** – `I'm a university year one student...` → Acknowledged, answers adapted
7. **Practice Exercises** – `I want to practice calculus for my final in math101...` → Generated exercises
8. **Geography Support** – Geography homework questions are now treated as in-scope subject requests
9. **Multi-turn Follow-up** – Follow-up questions maintain context coherence

## Architecture

```
User Input
    │
    ▼
┌─────────────────┐
│  Code Guardrail │──── reject obvious non-homework ───▶ Rejection Message
│  (regex/heuristic)│
└────────┬────────┘
         │ pass
         ▼
┌─────────────────┐
│ Conversation    │  ◄── add user message
│ Manager         │  ◄── build message list (system + history)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Client      │  Azure OpenAI / One API
│ (with retry)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Guardrail   │  System prompt enforces subject/homework rules
│ (system prompt) │
└────────┬────────┘
         │
         ▼
    Response to User
```

## License

This project is for academic purposes (HKUST CSIT5900 2025-26 Spring).
