"""
CONTROL ROOM: Shared State + Nodes + Edges + Conversation Memory.

The workflow has four routes.

START -> router
              |
              |-- notes_question -> retrieve_documents
              |                    -> generate_answer -> END
              |
              |-- task_or_math --> handle_task -> END
              |
              |-- planner -------> planner_node (MCP) -> END
              |
              +-- chitchat ------> handle_chat -> END

Conversation memory:
    SQLite stores LangGraph checkpoints.
    The history field stores previous chat messages.
    The checkpointer preserves state for each thread_id.

Streaming:
    The CLI uses graph.astream() with stream_mode="updates".
    Each finished node prints its name immediately,
    so you can watch the workflow move room by room.
"""

import asyncio
from typing import TypedDict

import aiosqlite
from pydantic import BaseModel, Field

from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from studybuddy.config import MEMORY_DB
from studybuddy.llm import get_llm
from studybuddy.retriever import search_notes
from studybuddy.router import router_node, route_decision
from studybuddy.tools import calculate
from studybuddy.mcp_client import planner_tools, text_nikalo


# ---------- SHARED GRAPH STATE ----------

class StudyState(TypedDict, total=False):
    """
    Shared state containing the information used by graph nodes.

    Each node reads the information it needs
    and returns the fields it wants to update.

    The history field stores previous chat conversations.
    """

    question: str
    route: str
    context: str
    sources: list
    answer: str
    history: list[tuple[str, str]]


# ---------- NODE 1: RETRIEVE DOCUMENTS ----------

def retrieve_documents(state):
    """
    Retrieve the top 3 relevant document chunks
    and update the context and sources fields.
    """

    documents = search_notes(state["question"], k=3)

    if not documents:
        raise RuntimeError(
            "The vector store is empty! "
            "Run ingestion first: "
            "uv run python -m studybuddy.ingest"
        )

    context = "\n\n".join(
        document.page_content for document in documents
    )

    sources = sorted({
        document.metadata.get("source_file", "?")
        for document in documents
    })

    return {
        "context": context,
        "sources": sources
    }


# ---------- NODE 2: GENERATE ANSWER ----------

ANSWER_PROMPT = """You are StudyBuddy, a helpful assistant.
You are provided with relevant pieces of information from the user's notes.

Rules:
1. Answer using ONLY the provided NOTES.
2. Do not use your own external knowledge.
3. If the answer is not available in the notes, reply:
   "I don't know."
4. Always mention the source file name at the end of your answer.

NOTES:
{context}

QUESTION:
{question}
"""


def generate_answer(state):
    """
    Generate an answer using the retrieved document context.
    """

    answer = get_llm().invoke(
        ANSWER_PROMPT.format(
            context=state["context"],
            question=state["question"]
        )
    ).content

    return {"answer": answer}


# ---------- NODE 3: HANDLE TASKS AND MATHEMATICAL QUESTIONS ----------

TASK_PROMPT = """
You are StudyBuddy's task assistant.

You have access to a calculator tool named 'calculate'.

For mathematical calculations:
1. Use the calculator tool to perform the calculation.
2. Do not calculate complex mathematical expressions yourself.
3. After receiving the tool result, provide a clear and simple answer.

For non-mathematical tasks, respond appropriately.
"""


def handle_task(state):
    """
    Handle mathematical questions and other tasks using the calculator.
    """

    llm = get_llm().bind_tools([calculate])

    messages = [
        ("system", TASK_PROMPT),
        ("human", state["question"]),
    ]

    response = llm.invoke(messages)

    while response.tool_calls:

        messages.append(response)

        for tool_call in response.tool_calls:

            print("\n========== CALCULATOR DEBUG ==========")
            print("Calculator Tool Called!")
            print("Tool Name:", tool_call["name"])
            print("Tool Arguments:", tool_call["args"])

            result = calculate.invoke(tool_call["args"])

            print("Tool Result:", result)
            print("======================================\n")

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                )
            )

        response = llm.invoke(messages)

    return {
        "answer": response.content,
        "sources": []
    }


# ---------- NODE 4: HANDLE GENERAL CONVERSATION WITH MEMORY ----------

CHAT_PROMPT = """You are StudyBuddy.
Talk to the user in a friendly and natural way.

Use the previous conversation when it is relevant.
If the user refers to something discussed earlier, use that context.

Previous conversation:
{history}

Current user question:
{question}

Give a short and friendly response in one or two sentences.
"""


def handle_chat(state):
    """
    Handle greetings and general conversations.

    This node:
    1. Reads the previous conversation history.
    2. Adds the recent history to the LLM prompt.
    3. Generates a response.
    4. Saves the current question and answer in history.
    """

    # Step 1: Read previously stored conversation history.
    history = state.get("history", [])

    # Step 2: Select the last six conversation exchanges.
    recent_history = history[-6:]

    # Step 3: Convert history into readable text.
    previous_conversation = "\n".join(
        f"User: {question} | StudyBuddy: {answer}"
        for question, answer in recent_history
    )

    # Step 4: Prepare the prompt with previous conversation.
    prompt = CHAT_PROMPT.format(
        history=previous_conversation or "No previous conversation.",
        question=state["question"]
    )

    # Step 5: Generate the response.
    answer = get_llm().invoke(prompt).content

    # Step 6: Save the current question and answer.
    updated_history = history + [
        (state["question"], answer)
    ]

    return {
        "answer": answer,
        "sources": [],
        "history": updated_history
    }


# ---------- NODE 5: MCP PLANNER REQUEST MODEL ----------

class PlanRequest(BaseModel):
    """
    Defines the structured information required by the planner tool.
    The LLM extracts the topic and number of days from the user's question.
    """

    topic: str = Field(
        description="The topic for which the user wants a study plan."
    )

    days: int = Field(
        description="The number of days for the study plan, between 1 and 14."
    )


# ---------- NODE 6: MCP PLANNER PROMPT ----------

PLAN_PROMPT = """You are StudyBuddy's study planning assistant.

Extract the following information from the user's question:
1. The topic for which the user wants a study plan.
2. The number of days requested.

If the user does not mention the number of days, use 7 days.

The number of days must be between 1 and 14.

User question: {question}
"""


# ---------- NODE 7: MCP PLANNER NODE ----------

async def planner_node(state):
    """
    Create a study plan using the external MCP Planner Server.

    Steps:
    1. Extract the topic and number of days using the LLM.
    2. Connect to the MCP Planner Server through mcp_client.
    3. Discover the make_study_plan tool.
    4. Execute the MCP tool.
    5. Return the generated plan to the graph state.
    """

    # Step 1: Extract the topic and duration from the user's question.
    structured_llm = get_llm().with_structured_output(PlanRequest)

    plan_request = await structured_llm.ainvoke(
        PLAN_PROMPT.format(question=state["question"])
    )

    # Step 2: Get available tools from the MCP Planner Server.
    tools = await planner_tools()

    # Step 3: Find the required MCP tool by its name.
    planner_tool = next(
        (tool for tool in tools if tool.name == "make_study_plan"),
        None
    )

    if planner_tool is None:
        raise RuntimeError(
            "The make_study_plan tool was not found on the MCP server."
        )

    # Step 4: Execute the MCP tool with the extracted information.
    raw_plan = await planner_tool.ainvoke({
        "topic": plan_request.topic,
        "days": plan_request.days
    })

    # Step 5: Convert the MCP response into readable text.
    plan = text_nikalo(raw_plan)

    print("\n========== MCP PLANNER DEBUG ==========")
    print("Planner Topic:", plan_request.topic)
    print("Requested Days:", plan_request.days)
    print("MCP Tool Called:", planner_tool.name)
    print("=======================================\n")

    return {
        "answer": plan,
        "sources": []
    }


# ---------- BUILD THE LANGGRAPH WORKFLOW ----------

def build_graph(checkpointer=None):
    """
    Register all nodes and connect them using conditional edges.

    The optional checkpointer enables persistent conversation state.
    """

    workflow = StateGraph(StudyState)

    # Register all graph nodes.
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("handle_task", handle_task)
    workflow.add_node("handle_chat", handle_chat)
    workflow.add_node("planner_node", planner_node)

    # Every request starts at the router.
    workflow.add_edge(START, "router")

    # Route the question according to the router's decision.
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "notes_question": "retrieve_documents",
            "task_or_math": "handle_task",
            "chitchat": "handle_chat",
            "planner": "planner_node"
        }
    )

    # Connect the RAG workflow.
    workflow.add_edge("retrieve_documents", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # Connect the task, chat, and MCP planner workflows.
    workflow.add_edge("handle_task", END)
    workflow.add_edge("handle_chat", END)
    workflow.add_edge("planner_node", END)

    # Compile the graph with the optional checkpointer.
    return workflow.compile(checkpointer=checkpointer)


# ---------- APPLICATION ENTRY POINT WITH SQLITE MEMORY ----------

async def main():
    """
    Start StudyBuddy with persistent conversation memory.

    SQLite stores checkpoints for each conversation thread.
    The CLI streams node-by-node progress using graph.astream().
    """

    # Ensure the data directory exists.
    MEMORY_DB.parent.mkdir(parents=True, exist_ok=True)

    # Open the SQLite database connection.
    async with aiosqlite.connect(str(MEMORY_DB)) as conn:

        # Create the LangGraph SQLite checkpointer.
        saver = AsyncSqliteSaver(conn)

        # Initialize checkpoint tables when needed.
        await saver.setup()

        # Build the graph with persistent memory.
        graph = build_graph(checkpointer=saver)

        # Identify this conversation thread.
        config = {
            "configurable": {
                "thread_id": "user-1"
            }
        }

        print("StudyBuddy is running! Type 'exit' to quit.")

        while True:

            question = input("\nYour question: ").strip()

            if question.lower() in ("exit", "quit", "bye"):
                print("Goodbye!")
                break

            if not question:
                continue

            # Execute the workflow with streaming (node-by-node updates).
            # Each yielded update is a dict: {node_name: fields_it_filled}.
            answer = ""
            sources = []

            async for update in graph.astream(
                {"question": question},
                config,
                stream_mode="updates"
            ):
                for node_name, node_output in update.items():
                    print(f"   [node done] {node_name}")

                    if node_output and "answer" in node_output:
                        answer = node_output["answer"]

                    if node_output and node_output.get("sources"):
                        sources = node_output["sources"]

            print("\nAnswer:")
            print(answer)

            if sources:
                print("\nSources:", ", ".join(sources))


if __name__ == "__main__":
    asyncio.run(main())