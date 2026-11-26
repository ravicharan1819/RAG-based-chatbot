"""Offline tests with fake embeddings / LLM (no API key needed): pytest -q"""
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from rag.pipeline import RAGChatbot, format_context, split_documents

DOCS = [
    Document(page_content="The Eiffel Tower is in Paris. " * 60, metadata={"source": "a.pdf", "page": 0}),
    Document(page_content="Photosynthesis converts light into chemical energy. " * 40,
             metadata={"source": "b.pdf", "page": 2}),
]


def test_split_keeps_metadata():
    chunks = split_documents(DOCS, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 2
    assert all("source" in c.metadata and "page" in c.metadata for c in chunks)


def test_ask_returns_answer_and_sources():
    bot = RAGChatbot(
        embeddings=DeterministicFakeEmbedding(size=64),
        llm=FakeListChatModel(responses=["It is in Paris (page 1)."]),
        k=3,
    )
    assert bot.add_documents(DOCS) > 0
    out = bot.ask("Where is the Eiffel Tower?")
    assert out["answer"] == "It is in Paris (page 1)."
    assert len(out["sources"]) == 3
    assert len(bot.history) == 2


def test_context_format():
    txt = format_context(DOCS[:1])
    assert txt.startswith("[1] (a.pdf, page 1)")
