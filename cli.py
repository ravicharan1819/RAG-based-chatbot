"""Command-line version of the chatbot.

Usage:
    python cli.py path/to/document.pdf [more.pdf ...]
"""
import sys

from dotenv import load_dotenv

from rag.pipeline import RAGChatbot, gemini_chat, gemini_embeddings, load_pdf


def main():
    load_dotenv()
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    bot = RAGChatbot(embeddings=gemini_embeddings(), llm=gemini_chat())
    for path in sys.argv[1:]:
        n = bot.add_documents(load_pdf(path))
        print(f"Indexed {path}: {n} chunks")
    print("Ask questions (type 'exit' to quit).")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        result = bot.ask(q)
        print(f"\nBot: {result['answer']}")
        pages = sorted({(d.metadata.get("source"), d.metadata.get("page", 0) + 1) for d in result["sources"]})
        print("Sources: " + "; ".join(f"{s} p.{p}" for s, p in pages))


if __name__ == "__main__":
    main()
