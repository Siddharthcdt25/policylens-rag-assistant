# System Architecture: Policy Assistant

`policy-assistant` is an enterprise RAG + OCR system designed specifically for insurance policy contracts. It emphasizes transparent provenance, deterministic page-level citations (`[filename p.N]`), and proactive refusal of ungrounded queries without hiding behind third-party orchestration frameworks.

---

## 1. High-Level System Architecture

```
                  ┌────────────────────────────────────────────────────────┐
                  │                 USER INTERFACES                        │
                  │   Streamlit Web UI (ui/)  /  FastAPI REST API (app/)   │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         [ INGESTION PIPELINE ]                            [ QUERY & RETRIEVAL ]
         (Uploads & Documents)                             (User Questions)
                      │                                               │
                      ▼                                               ▼
     ┌───────────────────────────────────┐               ┌────────────────────────┐
     │ PyMuPDF Page Text Extraction      │               │ Embed User Question    │
     │ Native font stream extraction     │               │ (sentence-transformers)│
     └────────────────┬──────────────────┘               └───────────┬────────────┘
                      │                                              │
         (Chars < 50) │ (Chars >= 50)                                │ Query Vector
         ┌────────────┴─────────────┐                                ▼
         ▼                          ▼                    ┌────────────────────────┐
┌──────────────────┐      ┌──────────────────┐           │ FAISS IndexFlatIP      │
│ Tesseract OCR    │      │ Native Text      │           │ Top-K Exact Search     │
│ (PyMuPDF Pixmap) │      │ Retained         │           └───────────┬────────────┘
└────────┬─────────┘      └────────┬─────────┘                       │
         │                         │                                 ▼
         └────────────┬────────────┘                     ┌────────────────────────┐
                      ▼                                  │ Check Similarity Score │
     ┌───────────────────────────────────┐               │ Top Score < Threshold? │
     │ Section-Aware Chunking            │               └───────────┬────────────┘
     │ RegEx Heading & Clause Detection  │                           │
     │ Target: ~400 tok, 15% Overlap     │             ┌─────────────┴─────────────┐
     └────────────────┬──────────────────┘             │ YES                       │ NO
                      ▼                                ▼                           ▼
     ┌───────────────────────────────────┐    ┌─────────────────┐        ┌─────────────────────┐
     │ L2 Normalized Vectorization       │    │ Direct Refusal  │        │ Assemble Prompt     │
     │ sentence-transformers/all-MiniLM  │    │ in Code (0 LLM  │        │ Strict Grounding    │
     └────────────────┬──────────────────┘    │ Tokens Bypassed)│        │ Cites [doc p.N]     │
                      ▼                       └─────────────────┘        └──────────┬──────────┘
     ┌───────────────────────────────────┐                                          │
     │ FAISS IndexFlatIP + JSON Metadata │                                          ▼
     │ index.faiss + metadata.json       │                               ┌─────────────────────┐
     └───────────────────────────────────┘                               │ Swappable LLM       │
                                                                         │ Gemini / Ollama     │
                                                                         └─────────────────────┘
```

---

## 2. Ingestion Pipeline Details

### Step 1: Hybrid Parser & Conditional OCR (`app/ingestion/parser.py`, `ocr.py`)
- **Native PDF Extraction**: PyMuPDF (`fitz`) parses text streams in C, running in sub-milliseconds per page with 100% font accuracy.
- **OCR Trigger**: Scanned riders, signed addenda, or graphic endorsements often yield 0 to 20 characters of native text. If `len(text) < OCR_MIN_CHARS` (default: 50), the system renders the page to an in-memory 300 DPI RGB pixmap (`page.get_pixmap(dpi=300)`) and calls Tesseract OCR (`pytesseract.image_to_string`).
- **No Poppler Dependency**: Rendering through PyMuPDF's internal MuPDF C-engine eliminates external system dependencies like `poppler-utils` (`pdftoppm`).

### Step 2: Section-Aware Chunking (`app/ingestion/chunker.py`)
- Standard text splitters slice blindly at character boundaries, severing exclusions from deductibles.
- Our chunker uses regex patterns matching insurance conventions:
  - `^(SECTION|ARTICLE|CLAUSE|PART|SCHEDULE)\s+[0-9A-Z\.\-]+`
  - `^\d+\.\d+(\.\d+)?\s+[A-Z]`
  - Uppercase headings (e.g. `EXCLUSIONS`, `DUTIES AFTER LOSS`)
- Smaller clauses ($< 400$ tokens) remain unbroken to preserve conditionality. Larger sections are partitioned via a sliding window with 15% overlap.
- Each chunk preserves metadata: `source_filename`, 1-indexed `page_number`, `section_title`, and `token_count`.

---

## 3. Retrieval Engine & Vector Mathematics (`app/retrieval/`)

### L2 Normalization & Inner Product Equivalence
We use `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
For any text vector $v$, L2 normalization projects it onto the unit hypersphere:
$$\hat{v} = \frac{v}{\|v\|_2} \implies \|\hat{v}\|_2 = 1.0$$

The inner product of two unit-normalized vectors $\hat{u}$ and $\hat{v}$ is identically equal to their **Cosine Similarity**:
$$\langle \hat{u}, \hat{v} \rangle = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \cos(\theta)$$

### Why `faiss.IndexFlatIP`?
1. **Exhaustive Exact Search**: IndexFlatIP calculates exact inner products across all vectors without loss of precision or recall.
2. **Deterministic Scale**: Scores are bounded in $[-1.0, 1.0]$. For insurance text, scores above $0.35$ reliably represent semantic overlap.
3. **Decoupled JSON Metadata**: FAISS indices store only raw float32 vectors. We store chunk text, page provenance, and section names in a clean, human-auditable `metadata.json` rather than risky Python pickles.

---

## 4. LLM Abstraction & Hallucination Defense (`app/llm/`)

### Strategy Pattern for Providers
The application couples only to the abstract class `LLMClient`:
```python
class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...
```
- **GeminiClient**: Uses official `google-genai` SDK with `temperature=0.0`.
- **OllamaClient**: Directly posts to Ollama's local HTTP endpoint (`/api/generate`) with `temperature=0.0` via `httpx`.

### Dual-Layer Refusal Mechanism
1. **Layer 1 (Code-Level)**: If $\max_{i}(\text{score}_i) < \text{threshold}$, the Python service returns `"Not found in the provided documents."` immediately. Zero LLM API calls, zero token cost, zero hallucination probability.
2. **Layer 2 (Prompt-Level)**: If the chunk passes threshold but lacks specific facts (e.g. asking about exotic exclusions not defined in the text), the system instructions strictly command the LLM to output `"Not found in the provided documents."`

---

## 5. Deployment Topologies

1. **Microservice / REST Architecture (Production)**:
   - FastAPI server (`app/main.py`) deployed via Docker on Cloud Run, Kubernetes, or AWS ECS.
   - Client applications consume `/ingest` and `/ask` with JSON payloads.
2. **Standalone Web Application (Streamlit Cloud)**:
   - The Streamlit interface (`ui/streamlit_app.py`) imports the retrieval modules directly in-process.
   - Enables instant zero-infrastructure deployment onto Streamlit Community Cloud with a single `requirements.txt` and `packages.txt`.
