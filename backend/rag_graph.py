from sqlalchemy import Boolean
from logger import setup_logger
from typing import Annotated, TypedDict, List, Dict, Literal, Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from openai import BaseModel
from langgraph.graph import StateGraph, END, add_messages
from rag import hybrid_search
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
load_dotenv()

logger = setup_logger(__name__)


class AgentState(TypedDict):
    question: str
    current_query: str

    retrieved_docs: List[Dict[str, Any]]   # raw retrieved docs
    relevant_docs: List[Dict[str, Any]]    # filtered docs

    retries: int
    decision: Literal["generate", "rewrite", "fallback"]

    answer: str
    sources: List[Dict[str, Any]]

    debug: Dict[str, Any]


SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": """Search the indexed PDF documents for passages relevant 
        to the query. Call this whenever you need information from the documents""",
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

llm = ChatOpenAI(model="gpt-4o-mini",
                 temperature=0).bind_tools([SEARCH_TOOL_SCHEMA])


def agent_node(state: AgentState) -> dict:
    """LLM decides whether to call search_documents or produce final answer."""

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
                    "source": d.metadata.get("source")
                })

            tool_messages.append(ToolMessage(
                content=result_text,
                tool_call_id=tool_call['id'],
                name="search_documents"
            ))

    return {
        "messages": tool_messages,
        "sources": state.get("sources", []) + new_sources,
    }


def should_continue(state: AgentState) -> str:
    """Route: if the agent emitted tool calls, go to tool. Otherwise end."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {
                            "tools": "tools", END: END})
graph.add_edge("tools", "agent")

react_agent = graph.compile()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4,
                                                         max=10))
def answer_question(question: str) -> tuple[str, list[dict]]:
    result = react_agent.invoke(
        {
            "messages": [{"role": "user", "content": question}],
            "sources": []
        }, config={"recursion_limit": 5}
    )

    answer = result["messages"][-1].content
    sources = result.get("sources", [])
    logger.info(f"ReAct answer generated, {len(sources)} source chunks used")

    return answer, sources
