# StudyBuddy AI Agent 🎓🤖

StudyBuddy is an intelligent, multi-route AI study assistant built with **LangGraph**, **ChromaDB (RAG)**, **Model Context Protocol (MCP)**, and **Groq LLM**. It features persistent conversation memory backed by **SQLite** checkpoints.

---

## 🚀 Features

- **Dynamic Routing:** Classifies user intent into study questions, study plan requests, mathematical calculations, or casual chitchat.
- **RAG (Retrieval-Augmented Generation):** Local document ingestion (`.pdf`, `.txt`, `.md`) with chunking, HuggingFace embeddings (`all-MiniLM-L6-v2`), and persistent vector storage in ChromaDB.
- **MCP Integration (Model Context Protocol):** Connects to a dedicated external Planner server via MCP stdio client for generating structured day-by-day study schedules.
- **AST-Safe Math Calculator:** Evaluates mathematical expressions securely using Python's Abstract Syntax Tree without unsafe `eval`.
- **Persistent Conversation Memory:** Utilizes `langgraph-checkpoint-sqlite` to maintain conversation thread state across sessions.

---

## 📁 Project Structure

```text
studybuddy/
├── data/
│   └── raw/                   # Source documents (PDF, TXT, MD)
├── src/
│   └── studybuddy/
│       ├── config.py          # Centralized settings & environment config
│       ├── graph.py           # LangGraph state machine & CLI interface
│       ├── ingest.py          # Document loader & ChromaDB ingestion pipeline
│       ├── llm.py            # Groq LLM factory
│       ├── mcp_client.py      # MCP client connector & cache
│       ├── planner_server.py  # FastMCP server for study planning
│       ├── retriever.py       # ChromaDB similarity search & retrieval
│       ├── router.py          # Intent classification & routing logic
│       └── tools.py           # Safe AST-based calculator tool
├── .env.example               # Template environment configuration
├── pyproject.toml             # Project dependencies & metadata
└── test_mcp.py                # MCP integration test script
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### 2. Install Dependencies
```bash
uv sync
```

### 3. Environment Variables
Create a `.env` file in the project root:
```bash
cp .env.example .env
```
Fill in your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📖 Usage

### Step 1: Ingest Documents (RAG)
Load and index raw documents from `data/raw/`:
```bash
uv run python -m studybuddy.ingest
```

### Step 2: Test Retrieval
Verify document chunks can be queried:
```bash
uv run python -m studybuddy.retriever
```

### Step 3: Run StudyBuddy CLI
Start the interactive conversational agent:
```bash
uv run python -m studybuddy.graph
```
