"""
Document Retrieval Module.

This module searches the ChromaDB vector store
and retrieves the most relevant document chunks
based on a user's question.
"""

from studybuddy.ingest import open_almari


def get_retriever(k=3):
    """
    Create and return a document retriever.

    Args:
        k (int): Number of relevant document chunks to retrieve.
                 Defaults to 3.

    Returns:
        A LangChain retriever configured for similarity search.
    """

    # Open the existing ChromaDB vector store.
    vector_store = open_almari()

    # Convert the vector store into a retriever.
    # The retriever will return the top-k relevant documents.
    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )


def search_notes(question, k=3):
    """
    Search the vector database for relevant document chunks.

    Args:
        question (str): The user's search query.
        k (int): Maximum number of relevant chunks to retrieve.

    Returns:
        list: A list of relevant LangChain Document objects.
    """

    # Create a retriever and execute the similarity search.
    return get_retriever(k=k).invoke(question)


# Execute the retrieval pipeline when this module
# is run directly from the command line.
if __name__ == "__main__":
    import sys

    # Ensure UTF-8 output encoding on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # Define a sample question for testing.
    question = "What is a router node?"

    # Display the search question.
    print(f"Question: {question}\n")

    # Retrieve the top 3 relevant document chunks.
    retrieved_chunks = search_notes(question, k=3)

    # Check whether any relevant documents were retrieved.
    if not retrieved_chunks:
        raise RuntimeError(
            "No relevant documents were retrieved. "
            "Make sure documents are ingested first using: "
            "uv run python -m studybuddy.ingest"
        )

    # Display each retrieved chunk with its source information.
    for index, chunk in enumerate(retrieved_chunks, start=1):

        # Get the original source file name from document metadata.
        source_file = chunk.metadata.get("source_file", "Unknown")

        # Get the PDF page number, if available.
        page_number = chunk.metadata.get("page")

        # Build the source location information.
        source_location = (
            f"{source_file} (Page {page_number})"
            if page_number is not None
            else source_file
        )

        print(
            f"--- Chunk #{index} | Source: {source_location} ---"
        )

        print(chunk.page_content[:200])

        print()
