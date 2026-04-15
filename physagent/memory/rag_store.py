"""RAG knowledge base: vectorstore management and RAG chain factory.

Refactored from RAG.py. Dropped: interactive Q&A loop, cloud search/download flow.
"""
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from physagent.config import DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K


def _get_embeddings():
    """Load the embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _get_indexed_sources(vectorstore):
    """Get set of already-indexed file basenames from the vectorstore."""
    try:
        collection = vectorstore._collection
        all_metadata = collection.get(include=["metadatas"])
        sources = set()
        for meta in all_metadata["metadatas"]:
            if meta and "source" in meta:
                sources.add(os.path.basename(meta["source"]))
        return sources
    except Exception:
        return set()


def _load_new_pdfs(data_dir: str, indexed_sources: set):
    """Load only PDFs not yet indexed."""
    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    all_docs = loader.load()
    return [
        doc for doc in all_docs
        if os.path.basename(doc.metadata.get("source", "")) not in indexed_sources
    ]


def build_vectorstore(data_dir: str, force_rebuild: bool = False) -> Chroma:
    """Build or incrementally update the vector knowledge base from a PDF directory.

    Args:
        data_dir: Path to directory containing PDF files.
        force_rebuild: If True, rebuild from scratch ignoring existing DB.

    Returns:
        Chroma vectorstore instance.
    """
    embeddings = _get_embeddings()

    # Incremental update if DB exists
    if os.path.exists(DB_DIR) and not force_rebuild:
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        if os.path.exists(data_dir):
            indexed = _get_indexed_sources(vectorstore)
            new_docs = _load_new_pdfs(data_dir, indexed)
            if new_docs:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
                )
                chunks = splitter.split_documents(new_docs)
                vectorstore.add_documents(chunks)
        return vectorstore

    # Full rebuild
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    if not docs:
        raise ValueError(f"No PDF content found in {data_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_DIR
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing ChromaDB vectorstore.

    Raises FileNotFoundError if DB doesn't exist.
    """
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError(
            f"No existing knowledge base at {DB_DIR}. "
            f"Run build_vectorstore() first."
        )
    embeddings = _get_embeddings()
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)


def get_rag_chain(vectorstore, llm, prompt_template: str):
    """Build a RAG retrieval chain.

    Args:
        vectorstore: Chroma vectorstore instance.
        llm: LangChain LLM instance.
        prompt_template: Prompt string with {context} and {question} placeholders.

    Returns:
        LCEL Runnable chain.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
