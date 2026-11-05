
"""
Central configuration module for the StudyBuddy application.

This module manages environment variables, LLM settings,
embedding model configuration, and storage paths.
"""

import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv


# Resolve the project root directory.
# config.py is inside src/studybuddy, so parents[2] points to the project root.
BASE_DIR = Path(__file__).resolve().parents[2]


# Load environment variables from the .env file
# before initializing the application settings.
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """
    Centralized application settings.

    frozen=True makes the settings object immutable,
    preventing accidental modification at runtime.
    """

    # LLM Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b"
    )

    # Embedding Model Configuration
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Data Storage Configuration
    DATA_RAW_DIR: str = os.getenv(
        "DATA_RAW_DIR",
        "data/raw"
    )

    CHROMA_PERSIST_DIR: str = os.getenv(
        "CHROMA_PERSIST_DIR",
        "data/vector_store"
    )

    CHROMA_COLLECTION_NAME: str = os.getenv(
        "CHROMA_COLLECTION_NAME",
        "studybuddy_notes"
    )


# SQLite database for persistent conversation memory.
MEMORY_DB = BASE_DIR / "data" / "memory.sqlite"


# Create a single shared settings instance
# that can be imported throughout the application.
settings = Settings()
