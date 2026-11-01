"""
Reranking service using FlashRank (CPU-only, ~67 MB, no GPU required).

Design notes:
- Model is loaded lazily on first call and cached as a module-level singleton,
  so all Uvicorn workers share state after the first warm-up (fork model).
- All public API follows the same dict format used by qdrant_service search methods.
- Always falls back to the original order if reranking fails, so the pipeline
  never breaks due to a reranker error.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Module-level singleton — loaded once, reused across all requests ──────────
_ranker = None


def _get_ranker():
    """Lazily load FlashRank ranker on first call; return cached instance after."""
    global _ranker
    if _ranker is not None:
        return _ranker

    try:
        import tempfile
        from pathlib import Path
        from flashrank import Ranker
        cache_path = str(Path(tempfile.gettempdir()) / "flashrank_cache")
        _ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=cache_path)
        logger.info("FlashRank reranker loaded (ms-marco-MiniLM-L-12-v2).")

    except ImportError:
        logger.warning(
            "flashrank is not installed — reranking disabled. "
            "Run: pip install flashrank"
        )

    except Exception as e:
        logger.warning("FlashRank failed to load: %s — reranking disabled.", e)

    return _ranker


def rerank(
    query: str,
    documents: List[Dict[str, Any]],
    top_n: int,
    score_threshold: float = 0.0,
) -> List[Dict[str, Any]]:

    """
    Rerank a list of retrieved document dicts by relevance to the query.

    Each document dict must have a 'content' key (same format returned by
    QdrantService.search_by_vector and hybrid_search_by_vector).

    Args:
        query:            The user's original query string.
        documents:        Candidate documents from vector search.
        top_n:            Maximum number of documents to return after reranking.
        score_threshold:  Drop documents with a relevance score below this value.
                          Defaults to 0.0 (keep everything above zero).

    Returns:
        List of document dicts sorted by relevance descending, capped at top_n.
        Returns original documents in original order if reranking fails.

    """
    if not documents:
        return documents

    ranker = _get_ranker()
    
    if ranker is None:
        # FlashRank unavailable — return top_n from original order as fallback
        return documents[:top_n]

    try:
        from flashrank import RerankRequest

        passages = [
            {"id": i, "text": doc.get("content", "")}
            for i, doc in enumerate(documents)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request) 

        # Filter by score threshold, then keep top_n
        filtered = [r for r in results if r.get("score", 0.0) >= score_threshold] 
        top_results = filtered[:top_n]
     
        # Map back to original document dicts preserving all metadata and attaching rerank score
        reranked_docs = []
        for r in top_results:
            doc = dict(documents[r["id"]])
            doc["rerank_score"] = float(r.get("score", 0.0))
            reranked_docs.append(doc) 

        logger.debug(
            "Reranked %d → %d docs (threshold=%.2f, top_n=%d)",
            len(documents),
            len(reranked_docs),
            score_threshold,
            top_n,
        )
        return reranked_docs

    except Exception as e:
        logger.warning("Reranking failed: %s — returning original top-%d docs.", e, top_n)
        return documents[:top_n]
