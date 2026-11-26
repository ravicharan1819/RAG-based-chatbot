"""Streamlit UI for the RAG document Q&A chatbot.

Run:  streamlit run app.py
"""
import os

import streamlit as st
from dotenv import load_dotenv

from rag.pipeline import RAGChatbot, gemini_chat, gemini_embeddings, load_uploaded_pdf

load_dotenv()
st.set_page_config(page_title="Document Q&A Chatbot", page_icon="📄", layout="wide")
st.title("📄 Chat with your PDFs")
st.caption("Retrieval-Augmented Generation with LangChain, FAISS and Google Gemini")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Gemini API key", value=os.getenv("GOOGLE_API_KEY", ""), type="password",
        help="Get a free key at https://aistudio.google.com/apikey",
    )
    k = st.slider("Chunks retrieved per question", 2, 10, 4)
    files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    process = st.button("Process documents", type="primary", disabled=not files)
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        if "bot" in st.session_state:
            st.session_state.bot.history.clear()

if process:
    if not api_key:
        st.sidebar.error("Please enter your Gemini API key.")
    else:
        bot = RAGChatbot(embeddings=gemini_embeddings(api_key), llm=gemini_chat(api_key), k=k)
        total = 0
        with st.spinner("Reading, chunking and embedding documents..."):
            for f in files:
                try:
                    total += bot.add_documents(load_uploaded_pdf(f.getvalue(), f.name))
                except ValueError as e:
                    st.sidebar.warning(f"{f.name}: {e}")
        if total:
            st.session_state.bot = bot
            st.session_state.messages = []
            st.sidebar.success(f"Indexed {len(files)} file(s) into {total} chunks.")

if "bot" not in st.session_state:
    st.info("Upload one or more PDFs in the sidebar and click **Process documents** to start.")
    st.stop()

st.session_state.bot.k = k
for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            try:
                result = st.session_state.bot.ask(question)
            except Exception as e:  # show API errors instead of crashing the app
                st.error(f"Error: {e}")
                st.stop()
        st.markdown(result["answer"])
        with st.expander("Sources"):
            for i, d in enumerate(result["sources"], 1):
                page = d.metadata.get("page")
                st.markdown(
                    f"**[{i}] {d.metadata.get('source')}"
                    + (f", page {page + 1}" if isinstance(page, int) else "") + "**"
                )
                st.caption(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
