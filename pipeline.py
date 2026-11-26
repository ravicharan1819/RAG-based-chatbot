"""RAG pipeline: load PDFs -> chunk -> embed -> vector similarity search -> Gemini answer."""
from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DEFAULT_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
DEFAULT_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions about the user's documents. "
            "Use ONLY the context below. If the answer is not in the context, say you could "
            "not find it in the documents. Keep answers concise and mention the page number "
            "when it helps.\n\nContext:\n{context}",
        ),
        ("placeholder", "{history}"),
        ("human", "{question}"),
    ]
)


def _pdf_to_documents(reader: PdfReader, source: str) -> list[Document]:
    """One Document per page; metadata keeps the file name and 0-based page number."""
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": source, "page": i}))
    return docs


def load_pdf(path: str | Path) -> list[Document]:
    """Load a PDF file from disk."""
    return _pdf_to_documents(PdfReader(str(path)), Path(path).name)


def load_uploaded_pdf(data: bytes, filename: str) -> list[Document]:
    """Load a PDF given as bytes (e.g. from Streamlit's file_uploader)."""
    return _pdf_to_documents(PdfReader(io.BytesIO(data)), filename)


def split_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    return [c for c in chunks if c.page_content.strip()]


def gemini_embeddings(api_key: str | None = None) -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=DEFAULT_EMBED_MODEL, google_api_key=api_key)


def gemini_chat(api_key: str | None = None, temperature: float = 0.2) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=DEFAULT_CHAT_MODEL, google_api_key=api_key, temperature=temperature
    )


def format_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page")
        page_txt = f", page {page + 1}" if isinstance(page, int) else ""
        parts.append(f"[{i}] ({d.metadata.get('source', 'document')}{page_txt})\n{d.page_content}")
    return "\n\n".join(parts)


@dataclass
class RAGChatbot:
    embeddings: Embeddings
    llm: BaseChatModel
    k: int = 4
    vectorstore: InMemoryVectorStore | None = None
    history: list[tuple[str, str]] = field(default_factory=list)

    def add_documents(self, docs: list[Document]) -> int:
        chunks = split_documents(docs)
        if not chunks:
            raise ValueError("No extractable text found (is the PDF scanned images only?).")
        if self.vectorstore is None:
            self.vectorstore = InMemoryVectorStore(self.embeddings)
        self.vectorstore.add_documents(chunks)
        return len(chunks)

    def retrieve(self, question: str) -> list[Document]:
        if self.vectorstore is None:
            raise RuntimeError("Upload and process at least one document first.")
        return self.vectorstore.similarity_search(question, k=self.k)

    def ask(self, question: str) -> dict:
        docs = self.retrieve(question)
        chain = PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke(
            {"context": format_context(docs), "question": question, "history": self.history[-6:]}
        )
        self.history += [("human", question), ("ai", answer)]
        return {"answer": answer, "sources": docs}

    def save(self, path: str | Path):
        """Save the index (chunks + embeddings) to a JSON file so documents need not be re-embedded."""
        if self.vectorstore is not None:
            self.vectorstore.dump(str(path))

    def load(self, path: str | Path):
        self.vectorstore = InMemoryVectorStore.load(str(path), self.embeddings)
