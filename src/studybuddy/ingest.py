
"""
Document Ingestion Pipeline.

This module loads PDF, TXT, and Markdown documents,
splits them into smaller chunks, generates embeddings,
and stores them in the ChromaDB vector database.
"""

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from studybuddy.config import settings


def load_documents():
    """
    Load all supported documents (PDF, TXT, MD)
    from the configured raw data directory.

    Each document is enriched with source file
    and file type metadata.
    """

    docs = []

    # Get the raw documents directory from application settings.
    raw_dir = Path(settings.DATA_RAW_DIR)

    # Return an empty list if the directory does not exist.
    if not raw_dir.exists():
        return docs

    # Iterate through all files in the raw data directory.
    for file in raw_dir.iterdir():

        # Get the file extension in lowercase.
        suffix = file.suffix.lower()

        # Select the appropriate document loader.
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file))

        elif suffix in (".txt", ".md"):
            loader = TextLoader(str(file), encoding="utf-8")

        else:
            # Skip unsupported file types.
            continue

        # Load document content and process its metadata.
        for document in loader.load():

            # Store the original file name in metadata.
            document.metadata["source_file"] = file.name

            # Store the document file type in metadata.
            document.metadata["file_type"] = suffix[1:]

            # Add the loaded document to the collection.
            docs.append(document)

    return docs


def split_documents(docs):
    """
    Split large documents into smaller text chunks.

    Chunk size: 400 characters.
    Chunk overlap: 60 characters.

    Overlapping helps preserve context between chunks.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", " ", ""],
    )

    # Split documents while preserving their metadata.
    return splitter.split_documents(docs)


def get_embedding_model():
    """
    Initialize and return the Hugging Face embedding model.

    The model converts text chunks into numerical vectors
    that can be stored and searched in ChromaDB.
    """

    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def open_almari():
    """
    Initialize or connect to the persistent ChromaDB vector store.

    Creates the configured collection if it does not exist.
    """

    return Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        embedding_function=get_embedding_model(),
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )


def chit_id(chit):
    """
    Generate a unique SHA-256 hash ID for a document chunk.

    The same text content always produces the same hash,
    helping prevent duplicate chunk IDs during ingestion.
    """

    return hashlib.sha256(
        chit.page_content.encode("utf-8")
    ).hexdigest()


def almari_count():
    """
    Return the total number of records stored
    in the configured ChromaDB collection.
    """

    almari = open_almari()

    return almari._collection.count()


def ingest_all_documents():
    """
    Execute the complete document ingestion pipeline.

    Steps:
    1. Load supported documents.
    2. Split documents into smaller chunks.
    3. Connect to the ChromaDB vector store.
    4. Generate unique IDs for chunks.
    5. Store unique chunks and their embeddings.
    6. Return ingestion statistics.
    """

    # Step 1: Load documents from the raw data directory.
    docs = load_documents()

    # Stop ingestion if no supported documents are found.
    if not docs:
        return {
            "status": "KHAALI",
            "detail": "No PDF, TXT, or Markdown files found in data/raw."
        }

    # Step 2: Split loaded documents into smaller chunks.
    chits = split_documents(docs)

    # Step 3: Initialize the ChromaDB vector store.
    almari = open_almari()

    # Step 4: Generate unique IDs and remove duplicate chunk content.
    unique_chits = {}

    for chit in chits:
        unique_chits[chit_id(chit)] = chit

    # Step 5: Store unique document chunks in ChromaDB.
    # The embedding function automatically generates embeddings.
    almari.add_documents(
        list(unique_chits.values()),
        ids=list(unique_chits.keys())
    )

    # Step 6: Return ingestion results and statistics.
    return {
        "status": "SUCCESS",
        "documents_loaded": len(docs),
        "chunks_created": len(chits),
        "chunks_added": len(unique_chits),
        "total_pins_in_almari": almari._collection.count(),
        "collection": settings.CHROMA_COLLECTION_NAME,
    }


# Execute the ingestion pipeline when this module is run directly.
if __name__ == "__main__":
    print(ingest_all_documents())
