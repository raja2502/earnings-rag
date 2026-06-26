# Earnings Intelligence

A RAG (Retrieval-Augmented Generation) application that lets you ask natural language questions across earnings call transcripts and financial statements from major companies — combining unstructured transcript text with structured financial tables in the same retrieval pipeline.

## Live Demo
[Try it here](https://earnings-rag-botaufvysvmmjpnazqfvar.streamlit.app/)

## Companies Covered
| Ticker | Company |
|--------|---------|
| JPM | JPMorgan Chase |
| GS | Goldman Sachs |
| BLK | BlackRock |
| RITM | Rithm Capital |
| MSFT | Microsoft |
| GOOGL | Google |
| AAPL | Apple |
| NVDA | NVIDIA |

## Quarters Covered
Q1 2025 · Q2 2025 · Q3 2025 · Q4 2025 · Q1 2026

## The Engineering Challenge
Standard RAG works well for unstructured text. Financial tables break when chunked naively — a number without its column header means nothing to an embedding model.

This app solves that by converting each financial row into a natural language sentence before embedding:

```
NVIDIA (NVDA) reported Revenue of 215938.0 in fiscal year ending 2026-01-31.
```

Combined with MMR (Maximum Marginal Relevance) retrieval to ensure coverage across all quarters rather than returning redundant top-ranked chunks.

## Example Questions
- How did NVIDIA's revenue trend across quarters?
- Did BlackRock's commentary on market volatility match their actual AUM numbers?
- What did JPMorgan say about interest rates vs their actual net interest income?
- Which company mentioned AI the most across all earnings calls?
- How did Apple's guidance compare to their actual revenue performance?

## Stack
- **LangChain** — orchestration and retrieval chain
- **FAISS** — local vector store for semantic search
- **Sentence Transformers** — local embeddings (no API cost)
- **Groq (Llama 3.3 70B)** — fast, free LLM for generation
- **Streamlit** — UI and deployment

## Project Structure
```
earnings-rag/
├── app.py           # Streamlit UI
├── rag.py           # RAG pipeline — embeddings, retrieval, generation
├── ingest.py        # Data ingestion — transcripts and financials
├── data/
│   ├── transcripts/ # Earnings call transcripts (.txt)
│   ├── financials/  # Financial statements (.csv)
│   └── vectorstore/ # FAISS index (auto-generated)
├── requirements.txt
└── .env             # API keys (not committed)
```

## Local Setup

1. Clone the repo:
```bash
git clone https://github.com/yourusername/earnings-rag.git
cd earnings-rag
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Create a `.env` file:
```
GROQ_API_KEY="your_groq_key"
HF_TOKEN="your_hf_token"
```

4. Add your data files to `data/transcripts/` and `data/financials/`

5. Run the app:
```bash
streamlit run app.py
```

## How It Works

1. **Ingest** — transcripts are loaded as text, financial CSVs are converted row by row into natural language sentences
2. **Chunk** — transcripts are split into 800-token chunks with 100-token overlap; financial sentences are kept as-is
3. **Embed** — all chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` locally
4. **Store** — embeddings are stored in a FAISS index
5. **Retrieve** — MMR retrieval fetches the top 10 diverse chunks for each query
6. **Generate** — Groq's Llama 3.3 70B generates a grounded answer using only the retrieved context