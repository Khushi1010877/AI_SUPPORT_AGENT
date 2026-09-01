import os
import shutil
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from config import settings

_embeddings = None


class DocumentIngestionError(Exception):
    pass


def get_embeddings():
    return None


def get_llm():
    provider = settings.LLM_PROVIDER.lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.OPENAI_CHAT_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.ANTHROPIC_CHAT_MODEL,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.2,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.GROQ_CHAT_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0.2,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER}")


def _collection_name(user_id: int) -> str:
    return f"user_{user_id}"


def _persist_dir(user_id: int) -> str:
    path = os.path.join(settings.CHROMA_DIR, f"user_{user_id}")
    os.makedirs(path, exist_ok=True)
    return path


def get_vectorstore(user_id: int) -> Chroma:
    return Chroma(
        collection_name=_collection_name(user_id),
        persist_directory=_persist_dir(user_id),
    )


def ingest_pdf(file_path: str, user_id: int, filename: str) -> int:
    """Load a PDF, split into chunks, embed, and add to the user's vector store.
    Returns the number of chunks added."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise DocumentIngestionError(
            "This PDF contains no extractable text. Upload a text-based PDF or run OCR first."
        )

    # Attach clean metadata for citations
    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1  # 1-indexed

    vectorstore = get_vectorstore(user_id)
    vectorstore.add_documents(chunks)
    return len(chunks)


def delete_document(user_id: int, filename: str):
    """Remove all chunks belonging to a given source filename."""
    vectorstore = get_vectorstore(user_id)
    vectorstore._collection.delete(where={"source": filename})


def wipe_user_store(user_id: int):
    persist_dir = _persist_dir(user_id)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)


SYSTEM_PROMPT = (
    "You are a helpful, professional customer support assistant. "
    "Answer the user's question using ONLY the information in the retrieved context below. "
    "If the answer is not contained in the context, say you don't have that information "
    "in the uploaded documents and suggest the user contact human support or upload a relevant document. "
    "Be concise and friendly. Do not make up facts.\n\n"
    "Context:\n{context}"
)


def answer_query(
    user_id: int, query: str, chat_history: List[Tuple[str, str]]
) -> Tuple[str, List[LCDocument]]:
    """
    chat_history: list of (role, content) tuples, role in {"user", "assistant"}
    Returns (answer, source_documents)
    """
    llm = get_llm()
    vectorstore = get_vectorstore(user_id)
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.RETRIEVER_K})

    # Convert history to LangChain message objects
    lc_history = []
    for role, content in chat_history:
        if role == "user":
            lc_history.append(HumanMessage(content=content))
        else:
            lc_history.append(AIMessage(content=content))

    # Rephrase follow-up questions into standalone queries using history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given a chat history and the latest user question, which might reference "
            "context in the chat history, rewrite it as a standalone question. "
            "Do NOT answer it, just reformulate if needed, otherwise return it as is."
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    result = rag_chain.invoke({"input": query, "chat_history": lc_history})
    answer = result["answer"]
    source_docs = result.get("context", [])
    return answer, source_docs
