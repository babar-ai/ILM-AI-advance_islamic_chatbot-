"""
Main Ingestion Orchestrator for the Islamic AI Chatbot.
Loads documents, splits general text books into chunks, creates Qdrant collections,
and uploads vector embeddings across all query engines with batching and auto-resume.

When settings.HYBRID_SEARCH_ENABLED=True, this script:
  1. Creates each collection with BOTH a named dense vector config ("dense") AND a
     sparse vector config (settings.SPARSE_VECTOR_NAME) for BM25 hybrid search.
  2. Computes BM25 sparse vectors for every document chunk using fastembed's BM25 model.
  3. Uploads dense + sparse vectors together using qdrant_client.upsert() with PointStruct.
"""

import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


from pathlib import Path
from typing import List, Any, Optional
import uuid

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Add project root and vector_store folder to sys.path
vector_store_dir = Path(__file__).resolve().parent
project_root = vector_store_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(vector_store_dir) not in sys.path:
    sys.path.insert(0, str(vector_store_dir))

from utils.custom_logger import setup_logger
from utils.config import settings
from services.qdrant_service import QdrantService

from document_loader import preprocess_docs, split_documents

logger = setup_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Legacy wrapper functions (kept for backwards compatibility)
# ─────────────────────────────────────────────────────────────────────

def setup_qdrant_collection(client: Any, collection_name: str) -> None:
    """Delegates collection setup to QdrantService."""
    qdrant_service = QdrantService()
    qdrant_service.setup_collection(collection_name)


def embed_and_upload(
    client: Any,
    chunks: List[Document],
    collection_name: str,
    embeddings: Any,
) -> None:
    """Delegates batched embedding and upload to QdrantService."""
    qdrant_service = QdrantService()
    qdrant_service.embed_and_upload(chunks, collection_name, embeddings)


# ─────────────────────────────────────────────────────────────────────
# Hybrid ingestion: dense + sparse vectors uploaded together
# ─────────────────────────────────────────────────────────────────────

def hybrid_embed_and_upload(
    qdrant_service: QdrantService,
    chunks: List[Document],
    collection_name: str,
    dense_embeddings: Any,
    bm25_model: Any,
    batch_size: int = 200,
) -> None:
    """
    Embeds all chunks with both dense (OpenAI) and sparse (BM25) vectors,
    then upserts them into the Qdrant collection using PointStruct.

    Each point is stored with:
      - vector={"dense": <openai_embedding>, SPARSE_VECTOR_NAME: SparseVector(...)}
      - payload={"page_content": ..., "metadata": ...}

    Args:
        qdrant_service:   QdrantService instance for direct client access.
        chunks:           List of LangChain Documents to ingest.
        collection_name:  Target Qdrant collection name.
        dense_embeddings: OpenAI embeddings model instance.
        bm25_model:       fastembed BM25 sparse encoder instance.
        batch_size:       Number of documents per upload batch.
    """
    from qdrant_client.models import PointStruct, SparseVector

    total = len(chunks)
    if total == 0:
        logger.warning("No chunks to upload for '%s', skipping.", collection_name)
        return

    # Check how many points already exist (for resume support)
    info = qdrant_service.client.get_collection(collection_name)
    existing_count = info.points_count

    if existing_count >= total:
        logger.info(
            "Collection '%s' already fully uploaded (%d/%d). Skipping!",
            collection_name, existing_count, total,
        )
        return

    start_idx = existing_count
    if start_idx > 0:
        logger.info(
            "Resuming '%s': skipping first %d chunk(s), uploading remaining %d ...",
            collection_name, start_idx, total - start_idx,
        )

    # ── Extract all text contents for BM25 encoding ───────────────────
    all_texts = [chunk.page_content for chunk in chunks]

    logger.info("Computing BM25 sparse vectors for %d documents ...", total)
    # fastembed BM25 returns a generator of SparseEmbedding objects
    sparse_embeddings_gen = bm25_model.embed(all_texts, batch_size=batch_size)
    sparse_embeddings = list(sparse_embeddings_gen)
    logger.info("BM25 sparse vectors computed.")

    total_batches = (total - start_idx + batch_size - 1) // batch_size

    for batch_start in range(start_idx, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_chunks = chunks[batch_start:batch_end]
        batch_sparse = sparse_embeddings[batch_start:batch_end]
        batch_num = (batch_start - start_idx) // batch_size + 1

        logger.info(
            "Batch %d/%d — dense-embedding %d docs (chunks %d–%d / %d) ...",
            batch_num, total_batches,
            len(batch_chunks), batch_start + 1, batch_end, total,
        )

        # Dense embeddings for this batch via OpenAI
        batch_texts = [c.page_content for c in batch_chunks]
        dense_vecs = dense_embeddings.embed_documents(batch_texts)

        # Build PointStructs with named dense + sparse vectors.
        # Use the single `vector={}` dict (compatible with all qdrant-client versions):
        #   - dense vector stored under key "dense"
        #   - sparse vector stored under key settings.SPARSE_VECTOR_NAME (e.g. "sparse")
        # Both go into the same dict — qdrant-client resolves dense vs sparse by type.
        points = []
        for chunk, dense_vec, sparse_emb in zip(batch_chunks, dense_vecs, batch_sparse):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        settings.SPARSE_VECTOR_NAME: SparseVector(
                            indices=sparse_emb.indices.tolist(),
                            values=sparse_emb.values.tolist(),
                        ),
                    },
                    payload={
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                )
            )

        qdrant_service.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        logger.info("Uploaded batch %d/%d to '%s'.", batch_num, total_batches, collection_name)

    final_count = qdrant_service.client.get_collection(collection_name).points_count
    logger.info(
        "✅ Hybrid upload complete for '%s' — total points in Qdrant: %d",
        collection_name, final_count,
    )


BASE_DIR = Path(__file__).parent / "storage"

SOURCES = {
    "quran":                BASE_DIR / "quran",
    "hadith":               BASE_DIR / "hadith",
    "tafseer":              BASE_DIR / "tafsir",
    "general_islamic_info": BASE_DIR / "general islamic books",
}


# ─────────────────────────────────────────────────────────────────────
# Main ingestion entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant collections.")
    parser.add_argument(
        "--source",
        type=str,
        default="all",
        choices=list(SOURCES.keys()) + ["all"],
        help="Specific source to ingest (default: all). E.g. --source general_islamic_info",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreate collection before ingestion.",
    )
    args = parser.parse_args()

    qdrant_service = QdrantService()

    # ── Dense embedding model (OpenAI) ────────────────────────────────
    logger.info("Loading OpenAI embedding model: %s ...", settings.EMBEDDING_MODEL)
    dense_embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        openai_api_key=settings.OPENAI_API_KEY or None,
    )
    logger.info("OpenAI Embedding model ready.")

    # ── Sparse embedding model (fastembed BM25) ───────────────────────
    bm25_model: Optional[Any] = None
    if settings.HYBRID_SEARCH_ENABLED:
        try:
            from fastembed.sparse.bm25 import Bm25
            logger.info("Loading fastembed BM25 sparse encoder (Qdrant/bm25) ...")
            bm25_model = Bm25(model_name="Qdrant/bm25", language="english")
            logger.info("BM25 sparse encoder ready.")
        except ImportError:
            logger.error(
                "fastembed is not installed! Run: pip install fastembed\n"
                "Falling back to dense-only ingestion."
            )
            bm25_model = None

    summary: dict[str, dict] = {}
    sources_to_process = (
        {args.source: SOURCES[args.source]} if args.source != "all" else SOURCES
    )

    for engine_name, source_path in sources_to_process.items():
        collection_name = settings.COLLECTION_NAMES[engine_name]

        logger.info("=" * 50)
        logger.info("Engine: %s | Collection: %s", engine_name.upper(), collection_name)
        logger.info("=" * 50)

        raw_docs = preprocess_docs(str(source_path))

        # Split long text books; JSON records (Quran, Hadith, Tafsir) are already 1 doc per entry
        if engine_name == "general_islamic_info":
            chunks = split_documents(raw_docs)
            logger.info(
                "%d raw books → %d chunks (chunk_size=%d, overlap=%d)",
                len(raw_docs), len(chunks), settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
            )
        else:
            chunks = raw_docs
            logger.info("%d record(s) kept as 1-to-1 documents (no text splitting needed)", len(chunks))

        # ── Create collection (with sparse config if hybrid enabled) ──
        qdrant_service.setup_collection(
            collection_name,
            force_recreate=True if args.force_recreate else None,
        )

        # ── Upload: hybrid (dense + sparse) OR dense-only ─────────────
        if settings.HYBRID_SEARCH_ENABLED and bm25_model is not None:
            logger.info("Using HYBRID ingestion (dense + BM25 sparse) for '%s' ...", collection_name)
            hybrid_embed_and_upload(
                qdrant_service=qdrant_service,
                chunks=chunks,
                collection_name=collection_name,
                dense_embeddings=dense_embeddings,
                bm25_model=bm25_model,
                batch_size=settings.BATCH_SIZE,
            )
        else:
            # When HYBRID_SEARCH_ENABLED=True, the collection was created with a named
            # "dense" vector. We MUST pass vector_name="dense" so QdrantVectorStore
            # writes into the correct named field instead of the default unnamed vector.
            vec_name = "dense" if settings.HYBRID_SEARCH_ENABLED else None
            logger.info(
                "Using DENSE-ONLY ingestion for '%s' (vector_name=%s) ...",
                collection_name, vec_name,
            )
            qdrant_service.embed_and_upload(
                chunks, collection_name, dense_embeddings,
                vector_name=vec_name,
            )

        summary[engine_name] = {"raw": len(raw_docs), "chunks": len(chunks)}

    logger.info("=" * 50)
    logger.info("INGESTION COMPLETE")
    logger.info("%-25s %10s  %10s", "Engine", "Raw Docs", "Chunks")
    logger.info("-" * 50)
    for engine_name, stats in summary.items():
        logger.info("%-25s %10d  %10d", engine_name, stats["raw"], stats["chunks"])
    logger.info("=" * 50)
