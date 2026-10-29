"""FastAPI Application providing /ingest, /ask, and /health endpoints.

Architecture Design:
This backend exposes the complete RAG + OCR engine as a standard REST API.
It enforces the similarity threshold in Python code, preventing unnecessary LLM token spend
and eliminating hallucination risks when queries have no relevance to the indexed documents.
"""

from contextlib import asynccontextmanager
import logging
import re
from typing import List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.ingestion.parser import parse_pdf
from app.ingestion.chunker import chunk_parsed_pages
from app.retrieval.embedder import get_embedder
from app.retrieval.store import VectorStore, RetrievalResult
from app.llm import get_llm_client, format_grounded_prompt, REFUSAL_MESSAGE

# Setup structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singleton vector store instance loaded at server startup
vector_store = VectorStore(dimension=384)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load existing vector store on startup if present."""
    logger.info("Initializing Policy Assistant RAG engine...")
    loaded = vector_store.load(settings.VECTOR_STORE_DIR)
    if loaded:
        logger.info(f"Loaded {len(vector_store)} chunks into vector store.")
    else:
        logger.info("No existing vector store found. Waiting for /ingest requests.")
    yield
    logger.info("Shutting down Policy Assistant engine...")


app = FastAPI(
    title="Policy Assistant API",
    description="Insurance Policy RAG + OCR with page-level citations and similarity refusal.",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for Streamlit or web frontend consumers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request & Response Schemas ---

class AskRequest(BaseModel):
    """Query payload for the /ask endpoint."""
    question: str = Field(..., min_length=2, description="The user question to answer.")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of context chunks to retrieve.")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum cosine similarity cutoff.")
    backend: Optional[str] = Field(None, description="Override LLM backend: 'gemini' or 'ollama'.")


class ChunkCitation(BaseModel):
    """Metadata and excerpt of a retrieved chunk."""
    chunk_id: str
    source_filename: str
    page_number: int
    section_title: str
    score: float
    snippet: str


class AskResponse(BaseModel):
    """Grounded answer response with citations and provenance."""
    question: str
    answer: str
    refused: bool
    refusal_reason: Optional[str] = None
    citations: List[str]
    retrieved_chunks: List[ChunkCitation]
    llm_backend: str
    model_name: str


class IngestResponse(BaseModel):
    """Result of document ingestion."""
    status: str
    filename: str
    pages_parsed: int
    pages_ocr_triggered: int
    chunks_created: int
    total_vectors_indexed: int


class HealthResponse(BaseModel):
    """System health and operational parameters."""
    status: str
    indexed_chunks: int
    embedding_model: str
    llm_backend: str
    llm_model: str
    default_top_k: int
    default_threshold: float


# --- Helper citation extractor ---

CITATION_REGEX = re.compile(r"\[([a-zA-Z0-9_\-\.\s]+)\s+p\.(\d+)\]")


def extract_citations_from_text(text: str) -> List[str]:
    """Find all [filename p.N] citation references in the model's answer."""
    matches = CITATION_REGEX.findall(text)
    unique_citations = []
    seen = set()
    for filename, page in matches:
        cit = f"[{filename.strip()} p.{page}]"
        if cit not in seen:
            seen.add(cit)
            unique_citations.append(cit)
    return unique_citations


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return pipeline health and runtime configuration."""
    current_llm = get_llm_client()
    return HealthResponse(
        status="healthy",
        indexed_chunks=len(vector_store),
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        llm_backend=current_llm.provider_name,
        llm_model=current_llm.model_name,
        default_top_k=settings.TOP_K,
        default_threshold=settings.SIMILARITY_THRESHOLD,
    )


@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """Upload and parse an insurance policy PDF, apply OCR fallback if needed, and index chunks."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        # 1. Parse pages (with OCR fallback for scanned pages)
        pages = parse_pdf(content, filename=file.filename)
        ocr_count = sum(1 for p in pages if p.used_ocr)

        # 2. Section-aware chunking
        chunks = chunk_parsed_pages(pages)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Could not extract any readable text or chunks from this PDF.",
            )

        # 3. Vectorization with sentence-transformers
        embedder = get_embedder()
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts)

        # 4. Add to FAISS index & persist to disk
        vector_store.add_chunks(chunks, embeddings)
        vector_store.save(settings.VECTOR_STORE_DIR)

        return IngestResponse(
            status="success",
            filename=file.filename,
            pages_parsed=len(pages),
            pages_ocr_triggered=ocr_count,
            chunks_created=len(chunks),
            total_vectors_indexed=len(vector_store),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Ingestion failed for {file.filename}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(exc)}",
        )


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """Answer a user question grounded in retrieved policy chunks with strict refusal logic."""
    if len(vector_store) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vector store is empty. Please upload at least one policy PDF via /ingest first.",
        )

    top_k = request.top_k or settings.TOP_K
    threshold = request.threshold if request.threshold is not None else settings.SIMILARITY_THRESHOLD

    # 1. Vectorize query
    embedder = get_embedder()
    query_vec = embedder.embed_query(request.question)

    # 2. Retrieve top-k chunks from FAISS
    results: List[RetrievalResult] = vector_store.search(query_vec, top_k=top_k)

    # Convert retrieved chunks into API response format
    retrieved_chunk_items = [
        ChunkCitation(
            chunk_id=r.chunk.chunk_id,
            source_filename=r.chunk.source_filename,
            page_number=r.chunk.page_number,
            section_title=r.chunk.section_title,
            score=round(r.score, 4),
            snippet=r.chunk.text[:220].replace("\n", " "),
        )
        for r in results
    ]

    # Select LLM client
    llm = get_llm_client(backend=request.backend)

    # 3. CODE-LEVEL REFUSAL: Check if the top retrieved chunk passes the similarity threshold
    if not results or results[0].score < threshold:
        top_score = results[0].score if results else 0.0
        reason = f"Top cosine similarity score ({top_score:.4f}) is below the required threshold ({threshold:.2f})."
        logger.info(f"Refusing query '{request.question}': {reason}")

        return AskResponse(
            question=request.question,
            answer=REFUSAL_MESSAGE,
            refused=True,
            refusal_reason=reason,
            citations=[],
            retrieved_chunks=retrieved_chunk_items,
            llm_backend=llm.provider_name,
            model_name=llm.model_name,
        )

    # 4. GROUNDED GENERATION: Call LLM with strict context-only prompt
    prompt = format_grounded_prompt(request.question, results)

    try:
        raw_answer = llm.generate(prompt)
    except Exception as exc:
        logger.error(f"LLM generation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM backend ({llm.provider_name}) generation failed: {str(exc)}",
        )

    # 5. Extract citations found in the LLM answer
    extracted_citations = extract_citations_from_text(raw_answer)

    # If the LLM itself determined context was insufficient and outputted the refusal string
    is_refused = REFUSAL_MESSAGE.lower() in raw_answer.lower()

    return AskResponse(
        question=request.question,
        answer=raw_answer,
        refused=is_refused,
        refusal_reason="Model determined context insufficient." if is_refused else None,
        citations=extracted_citations,
        retrieved_chunks=retrieved_chunk_items,
        llm_backend=llm.provider_name,
        model_name=llm.model_name,
    )
