# 🛡️ Policy Assistant: RAG + OCR with Page-Level Citations

A production-grade Retrieval-Augmented Generation (RAG) and Optical Character Recognition (OCR) system tailored for insurance policy PDFs. Delivers verifiable answers with deterministic page citations (`[filename p.N]`) and proactive refusal of ungrounded questions.

Built from scratch in Python 3.11 **without LangChain or LlamaIndex**, providing 100% auditable retrieval mechanics for high-stakes compliance and legal domains.

---

## 📐 Architecture Diagram

```
[Insurance Policy PDF] ──> [PyMuPDF Native Text Parser]
                                   │
                     (Chars < 50?) │ (Chars >= 50)
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
            [Tesseract OCR]              [Native Text]
            (In-Memory 300 DPI)                  │
                     └─────────────┬─────────────┘
                                   ▼
                   [Section-Aware Chunker]
                   - Regex clause detection (SECTION, ARTICLE, 2.1)
                   - Target: ~400 tokens, ~15% overlap
                   - Retains: [filename, page_number, section]
                                   ▼
             [sentence-transformers (all-MiniLM-L6-v2)]
                   - L2 Normalized Unit Vectors
                                   ▼
               [FAISS Vector Store (IndexFlatIP)]
               - index.faiss + decoupled metadata.json
                                   │
 [User Question] ──> [Embedder] ───┘
                           │
                           ▼
                 [Similarity Filter]
               (Top Score < Threshold?)
               ┌───────────┴───────────┐
         (Yes) │                       │ (No)
               ▼                       ▼
      [Code-Level Refusal]   [Grounded Context Assembler]
     "Not found in docs"     - Injects [filename p.N] headers
     (0 Tokens Spent)                  ▼
                             [Swappable LLM Client]
                             - Google Gemini (google-genai)
                             - Local Ollama (llama3.2)
                             (Temperature = 0.0)
                                       ▼
                             [Grounded Answer + Citations]
```

---

## ✨ Key Features

1. **Hybrid Ingestion with Conditional OCR**: Instant C-level text extraction via PyMuPDF; automatically falls back to in-memory Tesseract OCR for scanned riders and signed endorsements ($< 50$ characters).
2. **Section-Aware Chunking**: Regex-driven clause splitting that anchors text to contract headings, ensuring complete legal clauses remain intact.
3. **Pure-Python Direct Retrieval**: FAISS `IndexFlatIP` over L2-normalized vectors mathematically computes exact **Cosine Similarity** ($\cos\theta$).
4. **Dual-Layer Refusal & Hallucination Defense**:
   - **Layer 1 (Code-Level)**: Refuses queries directly in Python if the top similarity score is below `SIMILARITY_THRESHOLD` ($0.35$), consuming zero LLM tokens.
   - **Layer 2 (Prompt-Level)**: Enforces strict adherence to context and commands the model to reply *"Not found in the provided documents."* if evidence is missing.
5. **Swappable LLM Architecture**: Seamless hot-swapping between Google Gemini and offline local Ollama via a single environment variable (`LLM_BACKEND`).
6. **Dual Deployment Modes**:
   - **FastAPI REST Service** (`/ingest`, `/ask`, `/health`) for microservice integration.
   - **Streamlit Web UI** directly coupled to the core modules for zero-infrastructure deployment on Streamlit Community Cloud.
7. **Empirical Evaluation Suite**: Automated benchmark reporting **Hit@1/3/5**, **Mean Reciprocal Rank (MRR)**, and side-by-side **chunk-size ablation**.

---

## 📂 Repository Structure

```text
policy-assistant/
├── app/
│   ├── config.py                 # Pydantic Settings loaded from .env
│   ├── main.py                   # FastAPI application (/ingest, /ask, /health)
│   ├── ingestion/
│   │   ├── ocr.py                # In-memory PyMuPDF rendering + Tesseract OCR
│   │   ├── parser.py             # Hybrid PDF parser with conditional OCR trigger
│   │   └── chunker.py            # Section-aware sliding-window chunker
│   ├── retrieval/
│   │   ├── embedder.py           # sentence-transformers with L2 normalization
│   │   └── store.py              # FAISS IndexFlatIP + JSON metadata persistence
│   └── llm/
│       ├── base.py               # Abstract LLMClient interface
│       ├── gemini_client.py      # Google Gen AI SDK client
│       ├── ollama_client.py      # Native HTTP Ollama client via httpx
│       └── prompts.py            # Strict grounding instructions & context builder
├── ui/
│   └── streamlit_app.py          # Interactive web UI with provenance inspector
├── eval/
│   ├── questions.json            # 10 ground-truth evaluation test cases
│   └── run_eval.py               # Hit@K, MRR, and chunk-size comparison script
├── scripts/
│   ├── create_sample_policy.py   # Generates 3-page synthetic insurance PDF (with OCR page)
│   ├── inspect_chunks.py         # Inspects parser and chunker outputs
│   ├── inspect_retrieval.py      # Inspects FAISS similarity scores and rankings
│   └── test_ask.py               # CLI runner for grounded QA and refusal
├── docs/
│   └── architecture.md           # In-depth architectural specifications
├── tests/
│   ├── test_ingestion.py         # Unit tests for parser, OCR trigger, and chunker
│   ├── test_retrieval.py         # Unit tests for FAISS index and normalization
│   └── test_api_and_refusal.py   # Unit tests for FastAPI endpoints and refusal
├── requirements.txt              # Pinned Python package versions
├── packages.txt                  # Linux OS dependencies for Tesseract OCR
├── .env.example                  # Template configuration environment variables
└── README.md
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites & OS Setup

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv tesseract-ocr tesseract-ocr-eng
```

#### macOS (Homebrew)
```bash
brew install tesseract
```

#### Windows (PowerShell)
1. Install Tesseract OCR via Winget:
   ```powershell
   winget install UB-Mannheim.TesseractOCR
   ```
2. Add `C:\Program Files\Tesseract-OCR` to your system `PATH`.

---

### 2. Virtual Environment & Dependencies

```bash
cd policy-assistant

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

---

### 3. Environment Configuration

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` to configure your preferred backend:

```ini
# Choose "gemini" or "ollama"
LLM_BACKEND=gemini

# If using Gemini:
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# If using Ollama (Local):
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

---

## 💻 Running the System

### A. Launch Streamlit Web UI
```bash
streamlit run ui/streamlit_app.py
```
Open `http://localhost:8501`. Use the sidebar to click **"Load Sample"** to instantly test with our synthetic 3-page policy.

### B. Launch FastAPI Backend Service
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger API documentation will be available at `http://localhost:8000/docs`.

### C. Run Command-Line Inspection Scripts
```bash
# 1. Inspect chunking & OCR fallback:
python scripts/inspect_chunks.py

# 2. Inspect FAISS retrieval rankings & similarity scores:
python scripts/inspect_retrieval.py "What is the deductible for solar panels?"

# 3. Test grounded answering and code-level refusal:
python scripts/test_ask.py "What is the liability limit for dwelling coverage?"
python scripts/test_ask.py "Does the policy cover damages from UFO crashes?"
```

### D. Run the Evaluation Benchmark
```bash
# Standard evaluation (Hit@1, Hit@3, Hit@5, MRR):
python eval/run_eval.py

# Comparative Chunk-Size Experiment (200 tokens vs 600 tokens):
python eval/run_eval.py --compare 200 600
```

### E. Run Unit Test Suite
```bash
pytest tests/ -v
```

---

## 🧠 Key Design Decisions (Interview Talking Points)

| Design Choice | What We Did | Why We Did It (Trade-Off Rationale) |
| :--- | :--- | :--- |
| **Direct Retrieval vs. LangChain** | Direct FAISS + sentence-transformers | Eliminates opaque abstractions, excessive dependencies, and debugging overhead; every vector operation is transparent. |
| **Inner Product vs. L2 Distance** | `faiss.IndexFlatIP` over L2-normalized vectors | Inner product of unit vectors equals Cosine Similarity, bounding scores in $[-1.0, 1.0]$. Makes threshold filtering intuitive. |
| **In-Memory PyMuPDF vs. pdf2image** | `page.get_pixmap(dpi=300)` | Avoids introducing `poppler-utils` as a system requirement; PyMuPDF's C engine renders directly into PIL memory. |
| **Section-Aware Chunking** | Regex-anchored clause splitting | Prevents splitting policy clauses across deductibles and exclusions, preserving legal context and hierarchy in metadata. |
| **Decoupled JSON Metadata** | `index.faiss` + `metadata.json` | Avoids Python `pickle` security vulnerabilities (arbitrary code execution) and allows transparent inspection of stored chunks. |
| **Dual-Layer Refusal** | Python threshold check + Prompt rule | Blocks ungrounded queries in code before calling the LLM, reducing latency to $< 4$ ms and eliminating API token costs. |

---

## ⚠️ Known Limitations

1. **Table Extraction**: Complex multi-column deductible tables with merged cells are parsed as plain text sequences. High-complexity insurance schedules benefit from table-specialized extractors (e.g., Camelot, PaddleOCR, or layout-aware transformers).
2. **Dense-Only Retrieval**: Very rare alphanumeric policy rider codes (e.g., `HO-04-92`) can suffer from semantic mismatch in dense embeddings, which is best solved with hybrid sparse-dense search (BM25 + FAISS).
3. **Single Vector Store Instance**: Currently indexes policies in an in-memory FAISS flat index; multi-tenant enterprise isolation requires tenant-partitioned vector indexing.

---

## 🛣️ Path to Production (Enterprise Roadmap)

To evolve `policy-assistant` into a Tier-1 enterprise production service:

1. **Hybrid Sparse + Dense Retrieval (Ensemble)**:
   - Combine dense FAISS vectors with sparse BM25 indexing using Reciprocal Rank Fusion (RRF) to capture both semantic intent and exact clause codes (`Clause 4.2.1`).
2. **Two-Stage Re-Ranking**:
   - Retrieve top 20 candidates with FAISS, then pass them through a cross-encoder model (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to compress down to the top 5 most contextually relevant passages.
3. **LayoutLM / TrOCR for Degraded Scans**:
   - Replace Tesseract with an end-to-end vision-language OCR model (like TrOCR or LayoutLMv3) for degraded microfiche documents and handwritten loss notices.
4. **Multi-Tenant Managed Vector Database**:
   - Migrate local FAISS files to a managed vector store (e.g., Cloud SQL with `pgvector`, Pinecone, or Qdrant) with tenant-level access control (RBAC).
5. **Streaming Responses & Asynchronous Ingestion**:
   - Use server-sent events (SSE) for token streaming on the FastAPI `/ask` endpoint and Celery/Redis background queues for large multi-hundred-page policy ingestion.
6. **Hallucination & Compliance Telemetry**:
   - Integrate evaluation tracing (e.g. OpenTelemetry, Ragas, TruLens) to log citation ground truth accuracy and refusal rates in production.
