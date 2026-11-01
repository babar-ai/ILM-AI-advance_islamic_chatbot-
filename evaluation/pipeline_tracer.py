"""
pipeline_tracer.py
------------------
A thin wrapper around LangGraphService that captures the full pipeline
state (retrieved contexts + generated answer) needed for RAGAS evaluation.

Ensures that the contexts passed to RAGAS are the exact post-reranking chunks
formatted with identical citations (Surah, Ayah, Hadith narrator, Book title, URLs)
as seen by the generator LLM in LangGraphService._build_context().
"""

import sys
import time
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Ensure project root is on path when run directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schemas.data_classes.langraph_state import LangGraphState
from services.langgraph_service import LangGraphService, _extract_source_url, _timed_node
from utils.custom_logger import setup_logger

logger = setup_logger(__name__)


def _format_context_chunk(source_type: str, doc: dict) -> str:
    """
    Formats a single document chunk with the exact metadata and citation details
    seen by the LLM in LangGraphService._build_context().

    This guarantees that RAGAS faithfulness does not falsely penalize citations,
    and context_precision evaluates the true ranked information.
    """
    s_type = source_type.lower()
    meta = doc.get("metadata", {}) or {}
    content = doc.get("content", "").strip()

    if "web" in s_type:
        title = doc.get("title") or meta.get("title", "Web Source")
        url = (doc.get("url") or meta.get("url", "")).strip()
        lines = [f"[WEB] Title: {title}", f"Content: {content}"]
        if url:
            lines.append(f"Source URL: {url}")
        return "\n".join(lines)

    if "quran" in s_type:
        arabic = (
            meta.get("arabic")
            or meta.get("arabic_text")
            or meta.get("text_arabic")
            or meta.get("ayah_ar")
            or meta.get("ar")
            or meta.get("verse_arabic")
            or meta.get("ayah_text")
            or meta.get("arabic_ayah")
            or ""
        )
        if not arabic:
            for v in meta.values():
                if isinstance(v, str) and any("\u0600" <= c <= "\u06ff" for c in v):
                    arabic = v
                    break

        surah = meta.get("surah", "")
        ayah = meta.get("ayah_number", meta.get("reference", ""))
        ref = meta.get("reference", f"{surah} {ayah}")
        ref_str = str(meta.get("reference", "")).strip()

        if ref_str and ":" in ref_str:
            quran_url = f"https://quran.com/{ref_str}"
        else:
            surah_val = meta.get("surah_number") or meta.get("surah", "")
            ayah_val = meta.get("ayah_number", "")
            quran_url = f"https://quran.com/{surah_val}:{ayah_val}" if (surah_val and ayah_val) else "https://quran.com"

        lines = [f"[QURAN] Surah: {surah} (Ayah {ayah})"]
        if arabic:
            lines.append(f"Arabic Ayah: {arabic}")
        lines.append(f"English Translation: {content}")
        lines.append(f"Reference: {ref}")
        if quran_url:
            lines.append(f"Source URL: {quran_url}")
        return "\n".join(lines)

    elif "hadith" in s_type:
        title = meta.get("title") or meta.get("book_name") or "Hadith Collection"
        narrator = meta.get("narrator", "")
        source_url = _extract_source_url(meta)
        lines = [f"[HADITH] Collection: {title}"]

        if narrator:
            lines.append(f"Narrator: {narrator}")
        lines.append(f"Hadith Text: {content}")

        if source_url:
            lines.append(f"Source URL: {source_url}")
        return "\n".join(lines)

    elif "tafsir" in s_type or "tafseer" in s_type:
        tafsir_name = (
            meta.get("En_tafsir_source")
            or meta.get("tafsir_source")
            or "Tafsir Commentary"
        )
        surah = meta.get("surah", "")
        ayah = meta.get("ayah_number", "")
        source_url = _extract_source_url(meta)
        lines = [f"[TAFSIR] Commentary: {tafsir_name}"]

        if surah or ayah:
            lines.append(f"Surah: {surah} (Ayah {ayah})")
        lines.append(f"Commentary Text: {content}")

        if source_url:
            lines.append(f"Source URL: {source_url}")
        return "\n".join(lines)

    else:
        book_name = meta.get("book_name") or meta.get("title") or "Islamic Knowledge Base"
        source_url = _extract_source_url(meta)
        lines = [f"[{source_type.upper()}] Book / Source: {book_name}", f"Content: {content}"]
        
        if source_url:
            lines.append(f"Source URL: {source_url}")
        return "\n".join(lines)


class PipelineTrace:
    """Holds captured data from a single pipeline run."""

    def __init__(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        sources_used: list[str],
        latency_ms: float,
        error: Optional[str] = None,
    ):
        self.question = question
        self.answer = answer
        self.contexts = contexts          # flat list of formatted context chunks for RAGAS
        self.sources_used = sources_used  # e.g. ["quran", "hadith"]
        self.latency_ms = latency_ms
        self.error = error

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "sources_used": self.sources_used,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


class TracingLangGraphService(LangGraphService):
    """
    Subclass of LangGraphService that captures the full LangGraphState
    after the pipeline runs — extracting the exact reranked documents
    and formatted contexts passed to the generator node.
    """

    def __init__(
        self,
        qdrant_service=None,
        openai_service=None,
        classification_executor=None,
        retrival_executor=None,
    ):
        from services.qdrant_service import QdrantService
        from services.openai_service import OpenAIService

        classification_executor = classification_executor or ThreadPoolExecutor(max_workers=4)
        retrival_executor = retrival_executor or ThreadPoolExecutor(max_workers=8)
        qdrant_service = qdrant_service or QdrantService()
        openai_service = openai_service or OpenAIService(qdrant_service=qdrant_service)

        super().__init__(
            qdrant_service=qdrant_service,
            openai_service=openai_service,
            classification_executor=classification_executor,
            retrival_executor=retrival_executor,
        )

        self._build_eval_graph()

    def _build_eval_graph(self):
        """Compile the measured LangGraph pipeline with an in-memory checkpointer."""
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver

        workflow = StateGraph(LangGraphState)
        workflow.add_node("rewrite_query",       _timed_node("rewrite", self._rewrite_query))
        workflow.add_node("embed_query",          _timed_node("embed", self._embed_query))
        workflow.add_node("classify_and_search", self._classify_and_search)
        workflow.add_node("parallel_retrieve",   _timed_node("retrieve", self._parallel_retrieve))
        workflow.add_node("rerank_documents",    _timed_node("rerank", self._rerank_documents))
        workflow.add_node("generate_response",   _timed_node("generate", self._generate_response))

        workflow.set_entry_point("rewrite_query")
        workflow.add_edge("rewrite_query",       "embed_query")
        workflow.add_edge("embed_query",          "classify_and_search")
        workflow.add_edge("classify_and_search", "parallel_retrieve")
        workflow.add_edge("parallel_retrieve",   "rerank_documents")
        workflow.add_edge("rerank_documents",    "generate_response")
        workflow.add_edge("generate_response",   END)

        self.graph = workflow.compile(checkpointer=MemorySaver())

    def query_with_trace(self, user_query: str) -> PipelineTrace:
        """
        Runs the full LangGraph pipeline and returns a PipelineTrace
        containing the post-reranked, fully formatted contexts and generated answer.
        """
        start = time.perf_counter()
        error = None

        try:
            initial_state = LangGraphState(user_query=user_query)
            config = {"configurable": {"thread_id": f"eval_{int(time.time() * 1000)}"}}

            # Handle execution whether inside an existing event loop or not
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                except ImportError:
                    pass
                final_state_dict = loop.run_until_complete(
                    self.graph.ainvoke(initial_state, config=config)
                )
            else:
                final_state_dict = asyncio.run(
                    self.graph.ainvoke(initial_state, config=config)
                )

            # Extract fields from returned state dict
            answer: str = final_state_dict.get("final_response", "")
            retrieved_docs: dict = final_state_dict.get("retrieved_documents", {})
            web_results: list = final_state_dict.get("web_search_results", [])
            sources_used: list = final_state_dict.get("required_sources", [])

            # Collect all reranked Qdrant documents across sources
            all_doc_items = []
            for source_type, docs in retrieved_docs.items():
                for doc in docs:
                    all_doc_items.append((source_type, doc))

            # If reranker scores are attached, sort globally by relevance descending
            has_rerank_scores = any("rerank_score" in d for _, d in all_doc_items)
            if has_rerank_scores:
                all_doc_items.sort(key=lambda item: item[1].get("rerank_score", 0.0), reverse=True)

            # Format each chunk matching the generator prompt context
            contexts: list[str] = [
                _format_context_chunk(st, doc)
                for st, doc in all_doc_items
                if doc.get("content")
            ]

            # Append web search results as formatted chunks
            for web_doc in web_results:
                if web_doc.get("content"):
                    contexts.append(_format_context_chunk("web", web_doc))

        except Exception as e:
            logger.error("Pipeline trace failed for query '%s': %s", user_query[:60], e)
            answer = ""
            contexts = []
            sources_used = []
            error = str(e)

        latency_ms = (time.perf_counter() - start) * 1000

        trace = PipelineTrace(
            question=user_query,
            answer=answer,
            contexts=contexts,
            sources_used=sources_used,
            latency_ms=latency_ms,
            error=error,
        )

        logger.info(
            "Pipeline trace complete | sources=%s | contexts=%d | latency=%.0fms",
            sources_used,
            len(contexts),
            latency_ms,
        )
        return trace
