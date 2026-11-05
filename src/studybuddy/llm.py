
"""
LLM Factory Module.

This module is responsible for creating and configuring
the LLM instance used throughout the StudyBuddy application.
"""

from langchain_groq import ChatGroq

from studybuddy.config import settings


def get_llm():
    """
    Creates and returns a configured Groq LLM instance.

    The model configuration and API key are loaded
    from the centralized application settings.
    """

    # Validate that the Groq API key is configured.
    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is missing! "
            "Please check the .env file in the project root."
        )

    # Create and return the Groq Chat Model.
    return ChatGroq(
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=0,
    )
