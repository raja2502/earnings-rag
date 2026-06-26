import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN", "")

TRANSCRIPT_DIR = Path("data/transcripts")
FINANCIAL_DIR = Path("data/financials")

COMPANIES = {
    "JPM": "JPMorgan Chase",
    "GS": "Goldman Sachs",
    "BLK": "BlackRock",
    "RITM": "Rithm Capital",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AAPL": "Apple",
    "NVDA": "NVIDIA"
}


# --- Step 1: Load transcripts as Documents ---
def load_transcripts() -> list[Document]:
    docs = []

    for file in TRANSCRIPT_DIR.glob("*.txt"):
        parts = file.stem.split("_")  # e.g. JPM_Q1_2025

        if len(parts) < 3:
            continue

        ticker, quarter, year = parts[0], parts[1], parts[2]
        company = COMPANIES.get(ticker, ticker)

        text = file.read_text(encoding="utf-8", errors="ignore")

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": "transcript",
                    "ticker": ticker,
                    "company": company,
                    "quarter": quarter,
                    "year": year,
                    "file": file.name
                }
            )
        )

    print(f"  Loaded {len(docs)} transcripts")
    return docs


# --- Step 2: Convert financial tables to natural language ---
def table_row_to_text(row: pd.Series, ticker: str, quarter: str, year: str) -> str:
    """
    Convert each financial row into readable text for embeddings.
    """

    company = COMPANIES.get(ticker, ticker)
    metric = None
    parts = []

    for col, val in row.items():
        if pd.isna(val):
            continue

        val_str = str(val).strip()

        if val_str in ["", "nan", "None"]:
            continue

        if col.lower() in ["metric", "metrics", "financial metric", "line item"]:
            metric = val_str
        else:
            parts.append(f"{col}: {val_str}")

    if not parts:
        return ""

    if metric:
        return f"{company} ({ticker}) reported {metric} with values {', '.join(parts)}."
    
    return f"{company} ({ticker}) reported financial values {', '.join(parts)} in {quarter} {year}."


def load_financials() -> list[Document]:
    docs = []

    for file in FINANCIAL_DIR.glob("*.csv"):
        ticker = file.stem.split("_")[0].upper()
        company = COMPANIES.get(ticker, ticker)

        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"  Could not read {file.name}: {e}")
            continue

        for _, row in df.iterrows():
            quarter, year = "Annual", "Unknown"

            text = table_row_to_text(row, ticker, quarter, year)

            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": "financials",
                            "ticker": ticker,
                            "company": company,
                            "file": file.name
                        }
                    )
                )

    print(f"  Loaded {len(docs)} financial rows as text")
    return docs


# --- Step 3: Chunk transcripts ---
def chunk_documents(docs: list[Document]) -> list[Document]:
    transcript_docs = [
        d for d in docs
        if d.metadata.get("source") == "transcript"
    ]

    financial_docs = [
        d for d in docs
        if d.metadata.get("source") == "financials"
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunked = splitter.split_documents(transcript_docs)

    print(f"  Chunked transcripts into {len(chunked)} pieces")

    all_docs = chunked + financial_docs

    print(f"  Total documents in vector store: {len(all_docs)}")

    return all_docs


# --- Step 4: Build FAISS vector store ---
def build_vectorstore(docs: list[Document]) -> FAISS:
    print("\nBuilding vector store...")

    if not docs:
        raise ValueError("No documents found. Add transcripts or financial CSVs before building the vector store.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("data/vectorstore")

    print("  Vector store saved to data/vectorstore/")

    return vectorstore


# --- Step 5: Load vector store ---
def load_vectorstore() -> FAISS:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        "data/vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )


# --- Step 6: Build RAG chain with Groq ---
def build_rag_chain(vectorstore: FAISS):
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
               "system",
"""
You are a financial analyst assistant with access to earnings call transcripts and financial statements from major companies.

Use only the context below to answer the user's question accurately.

Rules:
- Focus only on the company mentioned in the question. Ignore context from other companies.
- When referencing numbers, always mention the company name, metric, quarter, and year.
- Cover ALL quarters available in the context, not just the most prominent ones.
- If the answer combines transcript commentary and financial data, clearly connect both.
- If the answer is not available in the context, say you do not know.
- Do not make up numbers.

Context:
{context}
"""
            ),
            ("human", "{input}")
        ]
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 30,  # considers 30 candidates, returns 10 diverse ones
            "lambda_mult": 0.5  # 0 = max diversity, 1 = max relevance
        }
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=combine_docs_chain
    )

    return rag_chain


# --- Main: build everything ---
def initialize():
    print("Loading documents...")

    transcript_docs = load_transcripts()
    financial_docs = load_financials()

    all_docs = chunk_documents(transcript_docs + financial_docs)

    vectorstore = build_vectorstore(all_docs)

    return build_rag_chain(vectorstore)

if __name__ == "__main__":
    chain = initialize()

    test_questions = [
        "What did JPMorgan say about interest rates in their latest earnings call?",
        "How did NVIDIA's revenue trend across quarters?",
        "Did BlackRock's commentary on market volatility match their actual AUM numbers?",
    ]

    print("\n--- Test Queries ---")
    for q in test_questions:
        print(f"\nQ: {q}")
        result = chain.invoke({"input": q})
        print(f"A: {result['answer']}")
        sources = set(d.metadata.get("source") for d in result["context"])
        print(f"Sources used: {sources}")