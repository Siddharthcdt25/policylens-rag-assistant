"""Streamlit Web UI for Policy Assistant.

Architecture Design:
This Streamlit application imports the RAG, OCR, and LLM modules directly.
This dual-mode architecture means:
1. It can run in a standalone single-container environment (e.g., Streamlit Community Cloud)
   without needing a separate FastAPI backend server running.
2. In enterprise setups, the exact same core logic is served concurrently via FastAPI (app/main.py).
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_parsed_pages
from app.retrieval.embedder import get_embedder
from app.retrieval.store import VectorStore, RetrievalResult
from app.llm import get_llm_client, format_grounded_prompt, REFUSAL_MESSAGE
from app.llm.gemini_client import GeminiClient
from app.llm.ollama_client import OllamaClient
from app.main import extract_citations_from_text
from scripts.create_sample_policy import generate_sample_policy

# Page layout & styling
st.set_page_config(
    page_title="Policy Assistant - Insurance RAG + OCR",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_cached_embedder():
    """Cache the sentence-transformers model so it is loaded into memory only once."""
    return get_embedder()


def get_vector_store() -> VectorStore:
    """Retrieve or initialize the vector store in Streamlit session state."""
    if "vector_store" not in st.session_state:
        store = VectorStore(dimension=384)
        store.load(settings.VECTOR_STORE_DIR)
        st.session_state.vector_store = store
    return st.session_state.vector_store


def main():
    # --- Sidebar: Configuration & Controls ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("**Swappable LLM & RAG Parameters**")

        # 1. LLM Backend Selection
        backend_choice = st.selectbox(
            "LLM Backend",
            options=["gemini", "ollama"],
            index=0 if settings.LLM_BACKEND == "gemini" else 1,
            help="Select cloud API (Gemini) or local offline inference (Ollama).",
        )

        gemini_api_key = ""
        gemini_model = settings.GEMINI_MODEL
        ollama_url = settings.OLLAMA_BASE_URL
        ollama_model = settings.OLLAMA_MODEL

        if backend_choice == "gemini":
            gemini_api_key = st.text_input(
                "Gemini API Key",
                value=settings.GEMINI_API_KEY,
                type="password",
                help="Enter key or set GEMINI_API_KEY in .env",
            )
            gemini_model = st.text_input(
                "Gemini Model",
                value=settings.GEMINI_MODEL,
                help="Gemini model name (e.g. gemini-2.5-flash)",
            )
        else:
            ollama_url = st.text_input(
                "Ollama Base URL",
                value=settings.OLLAMA_BASE_URL,
            )
            ollama_model = st.text_input(
                "Ollama Model",
                value=settings.OLLAMA_MODEL,
                help="Model pulled in Ollama (e.g. llama3.2)",
            )

        st.divider()

        # 2. Hyperparameters
        st.subheader("🎯 Pipeline Parameters")
        top_k = st.slider(
            "Top-K Retrieval",
            min_value=1,
            max_value=10,
            value=settings.TOP_K,
            help="Number of relevant chunks to retrieve and inject into prompt.",
        )

        threshold = st.slider(
            "Similarity Refusal Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(settings.SIMILARITY_THRESHOLD),
            step=0.05,
            help="Queries with top score below this are refused in code before calling LLM.",
        )

        chunk_size = st.slider(
            "Target Chunk Size (tokens)",
            min_value=150,
            max_value=800,
            value=settings.CHUNK_SIZE,
            step=50,
            help="Target token capacity for section-anchored chunks.",
        )

        chunk_overlap = st.slider(
            "Chunk Overlap (tokens)",
            min_value=10,
            max_value=150,
            value=settings.CHUNK_OVERLAP,
            step=10,
            help="Sliding window overlap between adjacent sub-chunks (~15%).",
        )

        ocr_min_chars = st.slider(
            "OCR Fallback Trigger (chars)",
            min_value=10,
            max_value=150,
            value=settings.OCR_MIN_CHARS,
            step=10,
            help="Trigger Tesseract if native page text is below this character count.",
        )

        st.divider()

        # 3. Vector Store Controls
        store = get_vector_store()
        st.subheader("📦 Vector Store Status")
        st.metric("Total Indexed Chunks", len(store))

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Clear Index", use_container_width=True):
                store.clear()
                store.save(settings.VECTOR_STORE_DIR)
                st.success("Index reset!")
                st.rerun()

        with col_b:
            if st.button("Load Sample", use_container_width=True, help="Load 3-page synthetic policy"):
                sample_pdf = ROOT_DIR / "data" / "sample_policy.pdf"
                if not sample_pdf.exists():
                    generate_sample_policy(sample_pdf)
                with open(sample_pdf, "rb") as f:
                    pages = parse_pdf(f.read(), filename="sample_policy.pdf", min_chars_for_ocr=ocr_min_chars)
                chunks = chunk_parsed_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                embedder = get_cached_embedder()
                embeddings = embedder.embed_texts([c.text for c in chunks])
                store.clear()
                store.add_chunks(chunks, embeddings)
                store.save(settings.VECTOR_STORE_DIR)
                st.success(f"Indexed {len(chunks)} chunks!")
                st.rerun()

    # --- Main Screen ---
    st.title("🛡️ Policy Assistant")
    st.markdown(
        "**Production RAG + OCR System for Insurance Policies with Verifiable Page-Level Citations.**  \n"
        "*Direct FAISS retrieval + PyMuPDF/Tesseract hybrid parsing. Built without LangChain.*"
    )

    # 1. Document Upload Section
    with st.expander("📄 Document Ingestion (Upload PDF with OCR Fallback)", expanded=(len(store) == 0)):
        uploaded_file = st.file_uploader(
            "Upload an Insurance Policy PDF (Native digital or scanned image)",
            type=["pdf"],
            help="Pages with < 50 characters automatically trigger Tesseract OCR.",
        )

        if uploaded_file is not None:
            if st.button("Parse & Index Document", type="primary"):
                with st.spinner("Processing document: PyMuPDF parsing -> OCR fallback -> Chunking -> Vectorizing..."):
                    file_bytes = uploaded_file.read()
                    pages = parse_pdf(
                        file_bytes,
                        filename=uploaded_file.name,
                        min_chars_for_ocr=ocr_min_chars,
                    )

                    ocr_triggered_pages = [p.page_number for p in pages if p.used_ocr]
                    chunks = chunk_parsed_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                    embedder = get_cached_embedder()
                    texts = [c.text for c in chunks]
                    embeddings = embedder.embed_texts(texts)

                    store.add_chunks(chunks, embeddings)
                    store.save(settings.VECTOR_STORE_DIR)

                st.success(
                    f"Successfully indexed **{len(chunks)} chunks** from **{len(pages)} pages** of `{uploaded_file.name}`!"
                )
                if ocr_triggered_pages:
                    st.info(f"🔍 OCR fallback was automatically triggered on scanned page(s): {ocr_triggered_pages}")
                st.rerun()

    st.divider()

    # 2. Question Answering Section
    st.subheader("💬 Ask a Policy Question")

    # Suggested queries for rapid testing
    st.caption("Quick test questions:")
    suggested_cols = st.columns(4)
    suggested_q = None

    if suggested_cols[0].button("🏠 Dwelling Deductible"):
        suggested_q = "What is the deductible for Coverage A dwelling losses?"
    if suggested_cols[1].button("☀️ Solar Equipment (OCR)"):
        suggested_q = "What is the maximum limit and deductible for solar panels?"
    if suggested_cols[2].button("🌊 Sewer Backup"):
        suggested_q = "Is sewer backup water damage covered or excluded?"
    if suggested_cols[3].button("👽 Alien Invasions (Refusal)"):
        suggested_q = "Does this policy cover damage caused by extraterrestrial alien spaceships?"

    # Input form
    question_input = st.text_input(
        "Enter your question regarding coverage, deductibles, exclusions, or riders:",
        value=suggested_q if suggested_q else "",
        placeholder="e.g., What are the policyholder's duties after a loss event?",
    )

    ask_button = st.button("Ask Assistant", type="primary", use_container_width=True)

    if (ask_button or suggested_q) and question_input.strip():
        if len(store) == 0:
            st.error("Vector store is empty! Please upload a PDF above or click 'Load Sample' in the sidebar.")
            return

        with st.spinner("Embedding query & searching FAISS index..."):
            embedder = get_cached_embedder()
            query_vec = embedder.embed_query(question_input)
            results: list[RetrievalResult] = store.search(query_vec, top_k=top_k)

        if not results:
            st.warning("No matches found in the vector index.")
            return

        top_score = results[0].score

        # --- CODE-LEVEL REFUSAL CHECK ---
        if top_score < threshold:
            st.error("⛔ Query Refused (Below Similarity Threshold)")
            st.markdown(f"**Answer:** `{REFUSAL_MESSAGE}`")
            st.warning(
                f"**Refusal Reason:** The top retrieved cosine similarity score was **{top_score:.4f}**, "
                f"which is below your configured threshold of **{threshold:.2f}**. "
                "The LLM call was bypassed in code to prevent hallucination and eliminate token costs."
            )
        else:
            # Instantiate configured LLM
            if backend_choice == "gemini":
                llm = GeminiClient(api_key=gemini_api_key, model_name=gemini_model)
            else:
                llm = OllamaClient(base_url=ollama_url, model_name=ollama_model)

            with st.spinner(f"Generating grounded answer with {llm.provider_name.upper()} ({llm.model_name})..."):
                prompt = format_grounded_prompt(question_input, results)
                try:
                    answer = llm.generate(prompt)
                except Exception as exc:
                    st.error(f"LLM Generation Error: {exc}")
                    return

            citations = extract_citations_from_text(answer)
            is_refused = REFUSAL_MESSAGE.lower() in answer.lower()

            # Display Answer Card
            st.markdown("### 📋 Grounded Policy Answer")
            if is_refused:
                st.warning(f"**{REFUSAL_MESSAGE}** (Model verified insufficient contextual evidence)")
            else:
                st.success(answer)

            # Display Citation Badges
            if citations:
                st.markdown("**Verified Source Citations:**")
                citation_cols = st.columns(len(citations))
                for idx, cit in enumerate(citations):
                    citation_cols[idx].info(f"🔖 {cit}")

        # --- RETRIEVED CHUNKS AUDIT EXPANDER ---
        with st.expander(f"🔍 Inspect Retrieved Chunks & Similarity Scores (Top {len(results)})", expanded=True):
            st.markdown("*Inspect the exact text chunks provided to the retrieval pipeline:*")
            for idx, r in enumerate(results, start=1):
                c = r.chunk
                is_above = r.score >= threshold
                score_badge = "✅ ABOVE THRESHOLD" if is_above else "❌ BELOW THRESHOLD"

                st.markdown(
                    f"**Rank #{idx} | Similarity: `{r.score:.4f}`** ({score_badge})  \n"
                    f"📍 **Citation:** `[{c.source_filename} p.{c.page_number}]` | **Section:** *{c.section_title}* | **Tokens:** ~{c.token_count}"
                )
                st.progress(max(0.0, min(1.0, float(r.score))))
                st.code(c.text, language="text")
                st.divider()


if __name__ == "__main__":
    main()
