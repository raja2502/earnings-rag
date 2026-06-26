import pandas as pd
from pathlib import Path

from rag import load_vectorstore, build_rag_chain


EVAL_FILE = Path("data/eval_questions.csv")


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    return str(value).lower().strip()


def evaluate_retrieval(vectorstore, eval_rows, k: int = 6):
    """
    Evaluates whether the retriever is pulling the right company and source type.

    Metrics:
    - ticker_hit_rate: expected ticker appears in retrieved docs
    - source_hit_rate: expected source type appears in retrieved docs
    - combined_hit_rate: both expected ticker and source appear in retrieved docs
    """

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    results = []

    for row in eval_rows:
        question = row["question"]
        expected_ticker = row["expected_ticker"]
        expected_source = row["expected_source"]

        docs = retriever.invoke(question)

        retrieved_tickers = {
            d.metadata.get("ticker")
            for d in docs
            if d.metadata.get("ticker")
        }

        retrieved_sources = {
            d.metadata.get("source")
            for d in docs
            if d.metadata.get("source")
        }

        ticker_hit = expected_ticker in retrieved_tickers
        source_hit = expected_source in retrieved_sources
        combined_hit = ticker_hit and source_hit

        results.append({
            "question": question,
            "expected_ticker": expected_ticker,
            "expected_source": expected_source,
            "retrieved_tickers": ", ".join(sorted(retrieved_tickers)),
            "retrieved_sources": ", ".join(sorted(retrieved_sources)),
            "ticker_hit": ticker_hit,
            "source_hit": source_hit,
            "combined_hit": combined_hit,
        })

    return pd.DataFrame(results)


def evaluate_answers(chain, eval_rows):
    """
    Evaluates whether the final answer contains an expected keyword or phrase.

    This is a simple evaluation method. It does not prove the answer is perfect,
    but it helps catch obvious failures.
    """

    results = []

    for row in eval_rows:
        question = row["question"]
        expected_contains = normalize_text(row["expected_answer_contains"])

        try:
            result = chain.invoke({"input": question})
            answer = result.get("answer", "")
            source_docs = result.get("context", [])

            retrieved_tickers = {
                d.metadata.get("ticker")
                for d in source_docs
                if d.metadata.get("ticker")
            }

            retrieved_sources = {
                d.metadata.get("source")
                for d in source_docs
                if d.metadata.get("source")
            }

            answer_lower = normalize_text(answer)

            if expected_contains == "i do not know":
                answer_hit = (
                    "do not know" in answer_lower
                    or "don't know" in answer_lower
                    or "not available" in answer_lower
                    or "not in the context" in answer_lower
                    or "not provided" in answer_lower
                )
            else:
                answer_hit = expected_contains in answer_lower

            results.append({
                "question": question,
                "expected_answer_contains": row["expected_answer_contains"],
                "answer": answer,
                "answer_hit": answer_hit,
                "retrieved_tickers": ", ".join(sorted(retrieved_tickers)),
                "retrieved_sources": ", ".join(sorted(retrieved_sources)),
            })

        except Exception as e:
            results.append({
                "question": question,
                "expected_answer_contains": row["expected_answer_contains"],
                "answer": "",
                "answer_hit": False,
                "retrieved_tickers": "",
                "retrieved_sources": "",
                "error": str(e),
            })

    return pd.DataFrame(results)


def print_summary(retrieval_df: pd.DataFrame, answer_df: pd.DataFrame):
    total = len(retrieval_df)

    ticker_accuracy = retrieval_df["ticker_hit"].mean() if total else 0
    source_accuracy = retrieval_df["source_hit"].mean() if total else 0
    combined_accuracy = retrieval_df["combined_hit"].mean() if total else 0
    answer_accuracy = answer_df["answer_hit"].mean() if len(answer_df) else 0

    print("\n==============================")
    print("RAG EVALUATION SUMMARY")
    print("==============================")
    print(f"Total questions: {total}")
    print(f"Ticker retrieval accuracy:   {ticker_accuracy:.2%}")
    print(f"Source retrieval accuracy:   {source_accuracy:.2%}")
    print(f"Combined retrieval accuracy: {combined_accuracy:.2%}")
    print(f"Answer keyword accuracy:     {answer_accuracy:.2%}")
    print("==============================")


def show_failures(retrieval_df: pd.DataFrame, answer_df: pd.DataFrame):
    retrieval_failures = retrieval_df[~retrieval_df["combined_hit"]]
    answer_failures = answer_df[~answer_df["answer_hit"]]

    if not retrieval_failures.empty:
        print("\nRetrieval failures:")
        for _, row in retrieval_failures.iterrows():
            print("\nQuestion:", row["question"])
            print("Expected:", row["expected_ticker"], row["expected_source"])
            print("Retrieved tickers:", row["retrieved_tickers"])
            print("Retrieved sources:", row["retrieved_sources"])

    if not answer_failures.empty:
        print("\nAnswer failures:")
        for _, row in answer_failures.iterrows():
            print("\nQuestion:", row["question"])
            print("Expected contains:", row["expected_answer_contains"])
            print("Answer:", row["answer"][:700])


def main():
    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"Missing {EVAL_FILE}. Create data/eval_questions.csv first."
        )

    print("Loading evaluation questions...")
    eval_df = pd.read_csv(EVAL_FILE)
    eval_rows = eval_df.to_dict("records")

    print("Loading vector store...")
    vectorstore = load_vectorstore()

    print("Evaluating retrieval...")
    retrieval_df = evaluate_retrieval(vectorstore, eval_rows, k=6)

    print("Building RAG chain...")
    chain = build_rag_chain(vectorstore)

    print("Evaluating final answers...")
    answer_df = evaluate_answers(chain, eval_rows)

    output_dir = Path("data/eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_output = output_dir / "retrieval_eval_results.csv"
    answer_output = output_dir / "answer_eval_results.csv"

    retrieval_df.to_csv(retrieval_output, index=False)
    answer_df.to_csv(answer_output, index=False)

    print_summary(retrieval_df, answer_df)
    show_failures(retrieval_df, answer_df)

    print(f"\nSaved retrieval results to: {retrieval_output}")
    print(f"Saved answer results to: {answer_output}")


if __name__ == "__main__":
    main()
