# ReAct LangGraph Migration Guide

## What changes and why

Your current pipeline is **linear and fixed**:

```
question → hybrid_search (always once) → LLM → answer
```

A **ReAct** (Reasoning + Acting) setup makes the LLM an agent that decides *when* and *how many times* to search, and can refine its query between calls:

```
question → LLM (reason: do I need to search?) → search tool → LLM (reason: enough info?) → search again or answer
```

The LLM drives the loop. This handles multi-part questions, follow-up searches, and lets the agent reason about whether the retrieved context actually answers the question before committing to an answer.

---

## Files that change

| File | Change |
|---|---|
| `backend/rag_graph.py` | Full rewrite — this becomes the ReAct graph |
| `backend/main.py` | One-line import swap |
| `backend/rag.py` | Keep as-is — `hybrid_search` and `_load_vectorstore` become the tool implementation |

Nothing changes in: `ingest.py`, `schemas.py`, `logger.py`, `frontend/`, Docker/SAM config.

---

## 1. Rewrite `backend/rag_graph.py`

Replace the entire file with this:

```python
from typing import Annotated, TypedDict

from langchain_core.messages import ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from tenacity import retry, stop_after_attempt, wait_exponential

from logger import setup_logger
from rag import hybrid_search  # reuse existing hybrid search + vectorstore logic

logger = setup_logger(__name__)


# --- State ---

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    sources: list[dict]  # accumulated across all tool calls


# --- Tool definition (no decorator needed — we call it manually) ---

SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Search the indexed PDF documents for passages relevant to the query. Call this whenever you need information from the documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
}


# --- Nodes ---

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools([SEARCH_TOOL_SCHEMA])


def agent_node(state: AgentState) -> dict:
    """LLM decides whether to call search_documents or produce a final answer."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Execute tool calls from the last agent message, update sources."""
    last_message: AIMessage = state["messages"][-1]
    tool_messages = []
    new_sources = []

    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "search_documents":
            query = tool_call["args"]["query"]
            logger.info(f"ReAct search: {query!r}")

            docs = hybrid_search(query, k=5)
            result_text = "\n\n".join(d.page_content for d in docs)

            for d in docs:
                new_sources.append({
                    "content": d.page_content,
                    "page": d.metadata.get("page"),
                    "source": d.metadata.get("source"),
                })

            tool_messages.append(ToolMessage(
                content=result_text,
                tool_call_id=tool_call["id"],
                name="search_documents",
            ))

    return {
        "messages": tool_messages,
        "sources": state.get("sources", []) + new_sources,
    }


def should_continue(state: AgentState) -> str:
    """Route: if the agent emitted tool calls, go to tools. Otherwise end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# --- Graph ---

graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

react_graph = graph.compile()


# --- Public API (matches signature expected by main.py) ---

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def answer_question(question: str) -> tuple[str, list[dict]]:
    result = react_graph.invoke({
        "messages": [{"role": "user", "content": question}],
        "sources": [],
    })
    answer = result["messages"][-1].content
    sources = result.get("sources", [])
    logger.info(f"ReAct answer generated, {len(sources)} source chunks used")
    return answer, sources
```

### Why this structure

- **`AgentState`** carries both `messages` (the standard ReAct conversation) and `sources` (accumulated across all tool calls, returned to the API).
- **`tool_node`** is custom (not `langgraph.prebuilt.ToolNode`) so it can write to `sources` in state. The prebuilt `ToolNode` only handles messages.
- **`should_continue`** is the routing function that closes the loop — if the LLM called a tool, run it; if not, the LLM is done reasoning and we exit.
- **`@retry`** moves from `rag.py` to here since this is now the entry point.

---

## 2. Update `backend/main.py` — one line

```python
# Before
from rag import answer_question

# After
from rag_graph import answer_question
```

The rest of `main.py` is unchanged. `answer_question(question)` returns `(answer, sources)` in both implementations.

---

## 3. Model: you must use a tool-calling model

Your current code uses `gpt-5-nano`. **This model may not support tool/function calling.** ReAct requires the LLM to emit structured tool call payloads.

Use one of these instead:

| Model | Notes |
|---|---|
| `gpt-4o-mini` | Cheapest option that reliably supports tool calling. Recommended. |
| `gpt-4o` | More capable, more expensive |
| `gpt-4.1-mini` | Alternative if available on your account |

Update both `rag_graph.py` (the `agent_node` LLM) and anywhere else `gpt-5-nano` appears in `rag.py` if you call it directly.

---

## What happened to `rewrite_query` and `grade_documents`

These were in your old `rag_graph.py`. In a ReAct setup:

- **`rewrite_query`**: Not needed. The agent naturally refines its search query by reasoning between tool calls — it will issue a second `search_documents` call with a better query if the first result was insufficient.
- **`grade_documents`**: Optional. You could add it as a second tool (`grade_relevance`), but the agent will reason about whether the context answers the question before it stops calling tools. Start without it.

---

## Things to know

### The agent can search multiple times
By default there's no cap on iterations. Add `recursion_limit` to `react_graph.compile()` or `react_graph.invoke()` to prevent runaway loops:

```python
result = react_graph.invoke({...}, config={"recursion_limit": 10})
```

### The system prompt matters
Without a system prompt, the agent may answer from its own knowledge instead of calling the tool. Add one:

```python
from langchain_core.messages import SystemMessage

def answer_question(question: str):
    result = react_graph.invoke({
        "messages": [
            SystemMessage(content="You are a document assistant. Always call search_documents before answering. Only use information from the search results. If the answer is not in the results, say 'I don't know'."),
            {"role": "user", "content": question},
        ],
        "sources": [],
    })
    ...
```

### Rate limiting still applies
The `@limiter.limit("10/minute")` in `main.py` wraps the `POST /ask` endpoint, not the internal graph. No change needed there.

### Testing
Your existing `test_api.py` tests the HTTP layer and will continue to work. The `test_rag.py` tests call `answer_question` directly — they'll exercise the new graph once you swap the import in `main.py`.

Run from project root:
```bash
pytest backend/test/test_api.py
```

### Observability
Each `search_documents` call is logged at INFO level (`ReAct search: "query"`). The number of source chunks is logged when the answer is returned. The existing Prometheus metrics at `/metrics` track request counts and latency at the HTTP layer — no changes needed.
