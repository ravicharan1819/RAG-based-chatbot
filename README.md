# RAG-Based Document Q&A Chatbot

An end-to-end **Retrieval-Augmented Generation (RAG)** chatbot that answers natural-language questions about PDF documents you upload. Built with **Python, LangChain, Google Gemini, PyPDF and Streamlit**.

## How it works
```
PDF upload ──► PyPDF text extraction (per page)
           ──► RecursiveCharacterTextSplitter (1000-char chunks, 200 overlap)
           ──► Gemini embeddings ──► vector store
Question  ──► embed question ──► similarity search (top-k chunks)
           ──► prompt = question + retrieved context + chat history
           ──► Gemini chat model ──► answer + cited source pages
```
- **Document loading** – each page becomes a LangChain `Document` with its file name and page number, so answers can cite sources.
- **Chunking** – overlapping chunks keep sentences that cross chunk boundaries retrievable.
- **Embeddings + similarity search** – chunks are embedded with Gemini embeddings and searched by cosine similarity; the index can be saved to and loaded from JSON.
- **Grounded answers** – the prompt tells the model to answer only from the retrieved context and to say so when the answer isn't in the documents.
- **Conversation memory** – the last few turns are passed to the model so follow-up questions work.

## Tech stack
Python · LangChain (core, text-splitters, langchain-google-genai) · Google Gemini · PyPDF · Streamlit

## Project structure
```
rag-document-qa-chatbot/
├── app.py            # Streamlit chat interface
├── cli.py            # terminal version
├── rag/pipeline.py   # loading, chunking, embeddings, retrieval, generation
├── tests/            # offline tests using fake embeddings / LLM
├── requirements.txt
└── .env.example
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then paste your key into .env
```
Get a free Gemini API key at <https://aistudio.google.com/apikey>.

## Run
```bash
streamlit run app.py                 # web app: upload PDFs in the sidebar, then chat
python cli.py my_document.pdf        # terminal version
pytest -q                            # tests (no API key needed)
```
The Gemini model names can be changed in `.env` (`GEMINI_CHAT_MODEL`, `GEMINI_EMBED_MODEL`) if Google retires a model.


