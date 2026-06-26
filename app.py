import streamlit as st
from pathlib import Path
from rag import load_vectorstore, build_rag_chain, initialize

# --- Page config ---
st.set_page_config(
    page_title="Earnings Intelligence",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .source-chip {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 500;
        margin-right: 6px;
    }
    .chip-transcript { background: #EEEDFE; color: #3C3489; }
    .chip-financials { background: #E1F5EE; color: #085041; }
    .company-tag {
        background: #F1EFE8;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 12px;
        color: #5F5E5A;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)

# --- Session state init ---
if "query" not in st.session_state:
    st.session_state["query"] = ""
if "run_query" not in st.session_state:
    st.session_state["run_query"] = False
if "last_query" not in st.session_state:
    st.session_state["last_query"] = ""

# --- Header ---
st.title("📊 Earnings Intelligence")
st.markdown("Ask questions across **earnings call transcripts** and **financial statements** from major companies — 2025 through Q1 2026.")

# --- Load or build RAG chain ---
@st.cache_resource(show_spinner=False)
def get_chain():
    vectorstore_path = Path("data/vectorstore")
    if vectorstore_path.exists():
        with st.spinner("Loading knowledge base..."):
            vs = load_vectorstore()
            return build_rag_chain(vs)
    else:
        with st.spinner("Building knowledge base for the first time — this takes ~3 minutes..."):
            return initialize()

chain = get_chain()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Companies covered")
    companies = {
        "JPM": "JPMorgan Chase",
        "GS": "Goldman Sachs",
        "BLK": "BlackRock",
        "RITM": "Rithm Capital",
        "MSFT": "Microsoft",
        "GOOGL": "Google",
        "AAPL": "Apple",
        "NVDA": "NVIDIA"
    }
    for ticker, name in companies.items():
        st.markdown(f"`{ticker}` {name}")

    st.markdown("---")
    st.markdown("### Quarters covered")
    st.markdown("Q1 2025 · Q2 2025 · Q3 2025 · Q4 2025 · Q1 2026")

    st.markdown("---")
    st.markdown("### Try asking")
    example_questions = [
        "How did NVIDIA's revenue trend across quarters?",
        "What was BlackRock's AUM in their most recent quarter?",
        "What did JPMorgan report as total revenue in Q1 2025?",
        "What did Jensen Huang say about NVIDIA infrastructure ROI?",
        "What was Apple's revenue in Q3 2025?",
        "Did BlackRock's commentary on volatility match their actual AUM numbers?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state["query"] = q
            st.session_state["run_query"] = True
            st.rerun()

# --- Main query input ---
query = st.text_input(
    "Ask a question",
    value=st.session_state["query"],
    placeholder="e.g. Did NVIDIA's commentary on data center demand match their actual revenue?",
)

# --- Run query if button clicked or new query typed ---
should_run = query and (
    st.session_state["run_query"] or
    query != st.session_state["last_query"]
)

if should_run:
    st.session_state["run_query"] = False
    st.session_state["last_query"] = query

    with st.spinner("Analyzing transcripts and financials..."):
        result = chain.invoke({"input": query})

    answer = result["answer"]
    source_docs = result["context"]

    # --- Answer ---
    st.markdown("### Answer")
    st.write(answer)

    # --- Sources used ---
    st.markdown("### Sources used")
    source_types = set(d.metadata.get("source") for d in source_docs)
    for s in source_types:
        chip_class = "chip-transcript" if s == "transcript" else "chip-financials"
        st.markdown(
            f'<span class="source-chip {chip_class}">{"📝 Transcript" if s == "transcript" else "📈 Financials"}</span>',
            unsafe_allow_html=True
        )

    st.markdown("")

    # --- Source documents detail ---
    with st.expander("See source documents"):
        for i, doc in enumerate(source_docs):
            meta = doc.metadata
            ticker = meta.get("ticker", "")
            quarter = meta.get("quarter", "")
            year = meta.get("year", "")
            source = meta.get("source", "")
            st.markdown(
                f'<span class="company-tag">{ticker}</span>'
                f'<span class="company-tag">{quarter} {year}</span>'
                f'<span class="company-tag">{source}</span>',
                unsafe_allow_html=True
            )
            st.markdown(f"*{doc.page_content[:300]}...*")
            if i < len(source_docs) - 1:
                st.divider()