
"""
Router: Reads each question and decides which workflow should handle it.

Four possible routes:
- notes_question -> Retrieve information from notes and generate an answer.
- task_or_math   -> Handle calculations or tasks using tools.
- planner        -> Create a study plan using the external MCP planner server.
- chitchat       -> Handle greetings and normal conversation.
"""

from typing import Literal

from pydantic import BaseModel, Field

from studybuddy.llm import get_llm


class RouteDecision(BaseModel):
    """
    Defines the router's structured output.

    The router must select exactly one of the four available routes.
    """

    route: Literal[
        "notes_question",
        "task_or_math",
        "chitchat",
        "planner"
    ] = Field(
        description=(
            "Classify the user's question into one of these categories: "
            "notes_question, task_or_math, chitchat, or planner."
        )
    )


ROUTER_PROMPT = """
You are the StudyBuddy router.

Your job is NOT to answer the user's question.
Your only responsibility is to classify the question
and decide which workflow should handle it.

Follow these rules:

1. notes_question:
   Use this route when the user asks about a knowledge topic
   or information that may exist in the notes.

   Examples:
   - What is RAG?
   - Explain LangGraph.
   - What is MCP?
   - Explain ChromaDB.
   - What is Python?

2. task_or_math:
   Use this route when the user requests a calculation
   or a task that may require a tool.

   Examples:
   - What is 2 + 2?
   - Calculate 15% of 800.
   - Calculate 500 divided by 10.

3. planner:
   Use this route when the user requests a study plan,
   learning schedule, or day-by-day learning roadmap.

   Examples:
   - Create a 7-day study plan for RAG.
   - Make a Python learning schedule.
   - Create a 5-day plan to learn LangGraph.

4. chitchat:
   Use this route for greetings and normal conversation.

   Examples:
   - Hello!
   - How are you?
   - Who are you?
   - Thank you.

User question: {question}

Return only the classification using the required structured output.
"""


def router_node(state):
    """
    Router node:
    Reads the user's question, classifies it using the LLM,
    and stores the selected route in the LangGraph state.
    """

    llm = get_llm()

    structured_llm = llm.with_structured_output(RouteDecision)

    decision = structured_llm.invoke(
        ROUTER_PROMPT.format(question=state["question"])
    )

    return {"route": decision.route}


def route_decision(state) -> Literal[
    "notes_question",
    "task_or_math",
    "chitchat",
    "planner"
]:
    """
    Conditional routing function:
    Reads the selected route from the state
    and tells LangGraph which workflow should execute.
    """

    return state["route"]


if __name__ == "__main__":

    test_questions = [
        "What is RAG?",
        "What is 2 + 2?",
        "Hello! How are you?",
        "Create a 7-day study plan for RAG.",
    ]

    llm = get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    for question in test_questions:

        decision = structured_llm.invoke(
            ROUTER_PROMPT.format(question=question)
        )

        print(f"Question: {question}")
        print(f"Selected Route: {decision.route}")
        print("-" * 40)
