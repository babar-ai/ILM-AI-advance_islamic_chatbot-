
import asyncio
import contextvars
import inspect
import re
import time
from typing import Optional, List, Dict, Any

from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient

from schemas.data_classes.langraph_state import LangGraphState
from services.qdrant_service import QdrantService
from services.openai_service import OpenAIService
from utils.config import settings
from utils.custom_logger import setup_logger
from fastembed.sparse.bm25 import Bm25
from langsmith import traceable

from services import rerank_service


logger = setup_logger(__name__)


# How many candidates to fetch from Qdrant (3-4× the final limit gives the
# reranker enough choices without blowing up RAM or latency).
SOURCE_RETRIEVE_LIMITS = {
    "quran": 15,
    "hadith": 18,
    "tafseer": 15,
    "general_islamic_info": 20,
}

# Final number of documents kept per source AFTER reranking.
# When reranking is disabled these are used directly as retrieval limits.
SOURCE_FINAL_LIMITS = {
    "quran": 3,
    "hadith": 3,
    "tafseer": 2,
    "general_islamic_info": 3,
}


def _timed_node(stage: str, node: Any) -> Any:
    """Measure a graph node, log its wall time, and add it to graph state."""
    if inspect.iscoroutinefunction(node):
        async def timed_async(state: LangGraphState) -> dict:
            started = time.perf_counter()
            try:
                result = await node(state)
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000
                logger.info("Pipeline stage timing | stage=%s duration_ms=%.2f", stage, elapsed_ms)

            update = dict(result)
            timings = dict(state.get("stage_timings_ms") or {})
            timings[stage] = round(elapsed_ms, 2)
            update["stage_timings_ms"] = timings
            return update

        return timed_async

    def timed_sync(state: LangGraphState) -> dict:
        started = time.perf_counter()
        try:
            result = node(state)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info("Pipeline stage timing | stage=%s duration_ms=%.2f", stage, elapsed_ms)

        update = dict(result)
        timings = dict(state.get("stage_timings_ms") or {})
        timings[stage] = round(elapsed_ms, 2)
        update["stage_timings_ms"] = timings
        return update

    return timed_sync


def _extract_source_url(meta: dict) -> Optional[str]:
    """Extracts authentic source URL directly from document metadata payload."""
    if not isinstance(meta, dict):
        return None
    for key in (
        "En_source_url",
        "source_url",
        "url",
        "link",
        "source_link",
        "Tafsir_Source",
    ):
        val = meta.get(key)
        if val and isinstance(val, str) and val.strip().startswith(("http://", "https://")):
            return val.strip()
    return None



def _remove_sources_section(text: str) -> str:
    """Strips out redundant '### 📚 Sources & References' section from the final response."""
    if not text:
        return text
    header_pattern = r"(?:^|\n)(?:#{1,4}\s*)?(?:📚\s*)?Sources\s*(?:&|and)\s*References\s*:?\s*(?:\r?\n|$)"
    match = re.search(header_pattern, text, re.IGNORECASE)
    if not match:
        return text

    before = text[:match.start()].rstrip()
    after = text[match.end():]

    disclaimer_match = re.search(
        r"(?:^|\n)(?:[*_>\s]*)(?:And\s+Allah\s+knows\s+best|وَاللَّ?هُ\s*أَعْلَمُ|Please\s+(?:always\s+)?consult\s+qualified\s+scholars)[\s\S]*$",
        after,
        re.IGNORECASE,
    )
    if disclaimer_match:
        return f"{before}\n\n{disclaimer_match.group(0).strip()}"
    return before


class LangGraphService:
    """
    6-stage LangGraph pipeline with persistent chat history via Redis checkpointer:
        rewrite_query → embed_query → classify_and_search → parallel_retrieve
                      → rerank_documents → generate_response
    """

    def __init__(
        self,
        qdrant_service,
        openai_service,
        classification_executor,
        retrival_executor
        ):

        self.qdrant_service = qdrant_service                         # Each QdrantClient can maintain its own HTTP connection pool.
        self.openai_service = openai_service
        self.classification_executor = classification_executor
        self.retrival_executor = retrival_executor
 
        # Tavily is optional — gracefully disabled if key is missing
        self.tavily_client = None
        if settings.TAVILY_API_KEY:

            try:
                self.tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
                logger.info("Tavily web search client initialized.")

            except Exception as e:
                logger.warning("Tavily client failed to initialize: %s. Web search disabled.", e)
                
        else:
            logger.warning("TAVILY_API_KEY not set — web search is disabled.")

        # ── BM25 sparse encoder (fastembed) for hybrid search ─────────
        # Loaded once at startup and shared across all retrieval calls.
        # query_embed() is called per-request in _parallel_retrieve.

        self._bm25_model = None

        if settings.HYBRID_SEARCH_ENABLED:

            try:
                self._bm25_model = Bm25(model_name="Qdrant/bm25", language="english")
                logger.info("BM25 sparse encoder (fastembed) loaded for hybrid search.")

            except ImportError:
                logger.warning(
                    "fastembed is not installed — hybrid search disabled. "
                    "Run: pip install fastembed"
                )

            except Exception as e:
                logger.warning("Failed to load BM25 model: %s — hybrid search disabled.", e)

        # ── Redis checkpointer is created lazily in _create_graph() ───
        # AsyncRedisSaver requires an async context manager; we store the
        # compiled graph reference after setup in application.py lifespan.
        self.graph = None  # Will be set by setup_graph() called from lifespan


 
    # ─────────────────────────────────────────────────────────
    # Node 0 (NEW): Rewrite ambiguous follow-up queries
    # ─────────────────────────────────────────────────────────

    def _rewrite_query(self, state: LangGraphState) -> dict:
        """
        Node 0 — runs BEFORE classify_and_search.

        Purpose:
          Resolve follow-up questions like "Tell me more" or "What about fasting?"
          into fully standalone questions by using the prior conversation turns
          stored in state["chat_history"].

        Gate:
          If chat_history is empty (first message of a session) we skip the LLM
          call entirely and just copy user_query → standalone_query at zero cost.

        Why return a dict instead of the full state?
          LangGraph nodes can return a partial dict — only the keys present in the
          dict are merged into the state via their reducers.  Returning only
          {standalone_query: ...} leaves all other state fields untouched, which is
          exactly what we want: don't reset retrieved_documents, chat_history, etc.
        """
        user_query = state.get("user_query", "")
        chat_history = state.get("chat_history") or []
        
        if not chat_history:
            # First turn — no rewrite needed
            logger.info("[rewrite_query] No history — using original query.")
            return {"standalone_query": user_query}

        # Slice to last MAX_TURNS pairs to keep the rewrite prompt compact
        max_turns = settings.CHAT_HISTORY_MAX_TURNS
        recent_history = chat_history[-(max_turns * 2):]  # each turn = 2 entries (user + assistant)

        ctx = contextvars.copy_context()
        future = self.classification_executor.submit(
            ctx.run,
            self.openai_service.rewrite_query,
            user_query,
            recent_history,
        )

        standalone = future.result()
        logger.info("[rewrite_query] standalone_query='%s'", standalone[:80])
        return {"standalone_query": standalone}


    # ─────────────────────────────────────────────────────────
    # Node 1: Embed the standalone query
    # ─────────────────────────────────────────────────────────

    def _embed_query(self, state: LangGraphState) -> dict:
        query = state.get("standalone_query") or state.get("user_query", "")
        try:
            return {"query_embedding": self.openai_service.embed_query(query)}
        except Exception as exc:
            logger.error("Error embedding standalone query: %s", exc)
            return {"query_embedding": None, "error_message": str(exc)}


    # ─────────────────────────────────────────────────────────
    # Node 2: Classify user query + web search (concurrent)
    # ─────────────────────────────────────────────────────────

    def _classify_and_search(self, state: LangGraphState) -> dict:
        """
        Classifies the query and runs web search concurrently via ThreadPoolExecutor.

        Uses standalone_query (not raw user_query) so that follow-up questions
        that were rewritten in Node 0 produce a meaningful classification result.
        """
        query_to_use = state.get("standalone_query") or state.get("user_query", "")
        query_embedding = state.get("query_embedding")
        if not query_embedding:
            return {
                "web_search_results": [],
                "required_sources": ["general_islamic_info"],
                "classification_reasoning": "",
                "filters": None,
                "error_message": state.get("error_message") or "Query embedding missing",
            }

        classification_timing: Dict[str, float] = {}

        def classify_with_timing() -> dict:
            started = time.perf_counter()
            try:
                return self.openai_service.classify_query(query_to_use, query_embedding)
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000
                classification_timing["classify"] = round(elapsed_ms, 2)
                logger.info(
                    "Pipeline stage timing | stage=classify duration_ms=%.2f",
                    elapsed_ms,
                )

        try:
            # Web search and classification run concurrently (propagating tracing context).
            # Each concurrent task must have its own context copy; sharing a single Context across
            # threads causes "cannot enter context: is already entered".
            ctx_web = contextvars.copy_context()
            ctx_classify = contextvars.copy_context()
            web_future = self.classification_executor.submit(ctx_web.run, self._run_web_search, query_to_use)
            classify_future = self.classification_executor.submit(ctx_classify.run, classify_with_timing)

            # Collect web search results
            web_search_results = web_future.result()
            logger.info(f"Web search returned {len(web_search_results)} results")

            # Collect classification results
            classification_response = classify_future.result()

            required_sources = ["general_islamic_info"]
            classification_reasoning = ""
            filters = None

            if classification_response["status"] == "success":
                classification = classification_response["message"]
                required_sources = list(classification.required_sources)
                classification_reasoning = classification.reasoning

                if hasattr(classification, "filters") and classification.filters:
                    filter_dict = classification.filters.model_dump(exclude_none=True)
                    filters = filter_dict if filter_dict else None

                logger.info(f"Classification: sources={required_sources}, reasoning={classification_reasoning}, filters={filters}")
            else:
                logger.warning(f"Classification failed: {classification_response['message']}. Falling back to general.")

            return {
                "query_embedding": query_embedding,
                "web_search_results": web_search_results,
                "required_sources": required_sources,
                "classification_reasoning": classification_reasoning,
                "filters": filters,
                "stage_timings_ms": {
                    **(state.get("stage_timings_ms") or {}),
                    **classification_timing,
                },
            }

        except Exception as e:
            logger.error(f"Error in _classify_and_search: {e}")
            return {
                "stage_timings_ms": {
                    **(state.get("stage_timings_ms") or {}),
                    **classification_timing,
                },
                "required_sources": ["general_islamic_info"],
                "error_message": str(e),
            }



    # ─────────────────────────────────────────────────────────
    # Node 2: Parallel retrieval from all required sources
    # ─────────────────────────────────────────────────────────

    def _parallel_retrieve(self, state: LangGraphState) -> dict:

        """
        Node 2: Dispatches Qdrant searches for all required sources
        concurrently using the pre-computed query embedding.

        When settings.HYBRID_SEARCH_ENABLED=True, each collection search uses
        hybrid RRF fusion (dense semantic + BM25 sparse keyword vectors).
        The BM25 sparse vector for the user query is computed once here and
        reused for all collection searches in this node.

        Falls back to dense-only search if hybrid fails or is disabled.
        """
        query_embedding = state.get("query_embedding")

        if not query_embedding:
            logger.error("No query embedding available for retrieval")
            return {"error_message": "Query embedding missing"}

        required_sources = state.get("required_sources") or []
        if not required_sources:
            logger.warning("No required sources identified, defaulting to general_islamic_info")
            required_sources = ["general_islamic_info"]

        # Use standalone_query for BM25 (context-aware sparse vector)
        query_text = state.get("standalone_query") or state.get("user_query", "")
        filters = state.get("filters")
        logger.info(f"Starting parallel retrieval for sources: {required_sources}, filters: {filters}")

        # ── Compute BM25 sparse vector for the query once (reused for all sources) ──
        sparse_indices: List[int] = []
        sparse_values: List[float] = []
        use_hybrid = settings.HYBRID_SEARCH_ENABLED and self._bm25_model is not None

        if use_hybrid:
            try:
                sparse_results = list(self._bm25_model.query_embed(query_text))
                if sparse_results:
                    sparse_emb = sparse_results[0]
                    sparse_indices = sparse_emb.indices.tolist()
                    sparse_values = sparse_emb.values.tolist()
                    logger.info(
                        "BM25 sparse vector computed for query (%d non-zero terms).",
                        len(sparse_indices),
                    )
                else:
                    logger.warning("BM25 returned empty sparse vector — falling back to dense-only.")
                    use_hybrid = False
            except Exception as e:
                logger.warning("BM25 query encoding failed: %s — falling back to dense-only.", e)
                use_hybrid = False

        future_to_source = {}

        for source in required_sources:
            collection_name = settings.COLLECTION_NAMES.get(source)

            if not collection_name:
                logger.warning(f"No collection configured for source: {source}")
                continue

            # Oversample when reranking is enabled; otherwise use final limits directly.
            if settings.RERANKER_ENABLED:
                limit = SOURCE_RETRIEVE_LIMITS.get(source, 15)
            else:
                limit = SOURCE_FINAL_LIMITS.get(source, 5)
            qdrant_filter = self.qdrant_service.build_qdrant_filter(collection_name, filters)

            # Each concurrent task must have its own context copy
            ctx = contextvars.copy_context()

            if use_hybrid:
                # Hybrid search: dense + BM25 sparse via RRF
                future = self.retrival_executor.submit(
                    ctx.run,
                    self.qdrant_service.hybrid_search_by_vector,
                    collection_name,
                    query_embedding,
                    sparse_indices,
                    sparse_values,
                    limit,
                    qdrant_filter,
                )
            else:
                # Dense-only fallback
                future = self.retrival_executor.submit(
                    ctx.run,
                    self.qdrant_service.search_by_vector,
                    collection_name,
                    query_embedding,
                    limit,
                    qdrant_filter,
                )

            future_to_source[future] = (source, collection_name, qdrant_filter, limit, use_hybrid)

        # Collect results as they complete
        retrieved_documents: Dict[str, List[Dict[str, Any]]] = {}
        for future in as_completed(future_to_source):
            source, collection_name, qdrant_filter, limit, was_hybrid = future_to_source[future]

            try:
                documents = future.result()

                # If filtered search returned nothing, retry without metadata filters
                if not documents and qdrant_filter is not None:
                    logger.warning(
                        "Filtered search returned 0 docs for '%s'. Falling back to unfiltered search...",
                        source,
                    )
                    if was_hybrid:
                        documents = self.qdrant_service.hybrid_search_by_vector(
                            collection_name,
                            query_embedding,
                            sparse_indices,
                            sparse_values,
                            limit,
                            query_filter=None,
                        )
                    else:
                        documents = self.qdrant_service.search_by_vector(
                            collection_name,
                            query_embedding,
                            limit,
                            query_filter=None,
                        )

                retrieved_documents[source] = documents
                logger.info(f"Retrieved {len(documents)} documents from '{source}' (hybrid={was_hybrid})")

            except Exception as e:
                logger.error(f"Error retrieving from '{source}': {e}")
                retrieved_documents[source] = []

        return {"retrieved_documents": retrieved_documents}




    # ─────────────────────────────────────────────────────────
    # Node 2.5: Rerank retrieved documents (optional)
    # ─────────────────────────────────────────────────────────
    @traceable(name="rerank_documents", run_type="tool")
    def _rerank_documents(self, state: LangGraphState) -> dict:
        """
        Optional node: applies FlashRank cross-encoder reranking to the
        retrieved documents for each source, keeping only the top-K most
        relevant results before they are assembled into the LLM context.

        Returns updated retrieved_documents dict.
        Falls back silently to top-K truncation if FlashRank is unavailable.
        """
        if not settings.RERANKER_ENABLED:
            return {}   # No changes — leave state as-is

        retrieved_documents = dict(state.get("retrieved_documents") or {})
        user_query = state.get("standalone_query") or state.get("user_query", "")

        for source, documents in retrieved_documents.items():
            if not documents:
                continue

            top_n = SOURCE_FINAL_LIMITS.get(source, 5)

            reranked = rerank_service.rerank(
                query=user_query,
                documents=documents,
                top_n=top_n,
                score_threshold=settings.RERANKER_SCORE_THRESHOLD,
            )

            retrieved_documents[source] = reranked
            logger.info(
                "Reranked '%s': %d → %d docs", source, len(documents), len(reranked)
            )

        return {"retrieved_documents": retrieved_documents}


    # ─────────────────────────────────────────────────────────
    # Shared helper: Build context string from state
    # ─────────────────────────────────────────────────────────
    def _build_context(self, state: LangGraphState) -> str:
        """
        Compiles all retrieved documents + web search results into a single context string.
        Source links are dynamically extracted directly from the Qdrant metadata payload.
        """
        context_sections = []

        # 1. Add web search results with URLs
        web_search_results = state.get("web_search_results") or []
        retrieved_documents = state.get("retrieved_documents") or {}

        if web_search_results:
            web_context = "\n--- WEB SEARCH RESULTS ---\n"
            for i, doc in enumerate(web_search_results):
                title = doc.get("title", "Web Source")
                url = doc.get("url", "").strip()
                content = (doc.get("content", "") or "")[:800]
                web_context += (
                    f"{i+1}.\n"
                    f"Title: {title}\n"
                    f"Content: {content}\n"
                )
                DISALLOWED_DOMAINS = (
                    "facebook.com", "instagram.com", "twitter.com", "x.com",
                    "reddit.com", "pinterest.com", "tiktok.com", "youtube.com"
                )
                if url and not any(d in url.lower() for d in DISALLOWED_DOMAINS):
                    web_context += f"Source URL: {url}\n"
                web_context += "\n"
            context_sections.append(web_context)

        # 2. Add retrieved documents by source, extracting authentic source links directly from metadata
        for source_type, documents in retrieved_documents.items():
            if not documents:
                continue

            s_type = source_type.lower()
            source_context = f"\n--- {source_type.upper()} SOURCES ---\n"

            for i, doc in enumerate(documents):
                meta = doc.get("metadata", {})
                content = doc.get("content", "")

                # ── 1. Quran Collection: Direct verse link to https://quran.com/{surah}:{ayah} ──
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
                        # Fallback: scan metadata fields for Arabic characters
                        for k, v in meta.items():
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

                    source_context += (
                        f"{i+1}.\n"
                        f"Surah: {surah} (Ayah {ayah})\n"
                        f"Arabic Ayah: {arabic}\n"
                        f"English Translation: {content}\n"
                        f"Reference: {ref}\n"
                        f"Source URL: {quran_url}\n\n"
                    )

                # ── 2. Hadith Collection: Dynamic extraction from chunk metadata ──
                elif "hadith" in s_type:
                    title = meta.get("title") or meta.get("book_name") or "Hadith Collection"
                    narrator = meta.get("narrator", "")
                    source_url = _extract_source_url(meta)

                    source_context += (
                        f"{i+1}.\n"
                        f"Hadith Collection: {title}\n"
                        f"Narrator: {narrator}\n"
                        f"Hadith Text: {content}\n"
                    )
                    if source_url:
                        source_context += f"Source URL: {source_url}\n"
                    source_context += "\n"

                # ── 3. Tafsir Collection: Dynamic extraction from chunk metadata ──
                elif "tafsir" in s_type or "tafseer" in s_type:
                    tafsir_name = (
                        meta.get("En_tafsir_source")
                        or meta.get("tafsir_source")
                        or "Tafsir Commentary"
                    )
                    surah = meta.get("surah", "")
                    ayah = meta.get("ayah_number", "")
                    source_url = _extract_source_url(meta)

                    source_context += (
                        f"{i+1}.\n"
                        f"Tafsir Commentary: {tafsir_name}\n"
                        f"Surah: {surah} (Ayah {ayah})\n"
                        f"Commentary Text: {content}\n"
                    )
                    if source_url:
                        source_context += f"Source URL: {source_url}\n"
                    source_context += "\n"

                # ── 4. General Islamic Info: Dynamic extraction from chunk metadata ──
                else:
                    book_name = meta.get("book_name") or meta.get("title") or "Islamic Knowledge Base"
                    source_url = _extract_source_url(meta)

                    source_context += (
                        f"{i+1}.\n"
                        f"Book / Source: {book_name}\n"
                        f"Content: {content}\n"
                    )
                    if source_url:
                        source_context += f"Source URL: {source_url}\n"
                    source_context += "\n"

            context_sections.append(source_context)

        full_context = "\n\n".join(context_sections)
        # Enforce budget to ensure total tokens fit within LLM rate limits (e.g. Groq 8000 TPM)
        MAX_TOTAL_CONTEXT_CHARS = 12000
        if len(full_context) > MAX_TOTAL_CONTEXT_CHARS:
            full_context = full_context[:MAX_TOTAL_CONTEXT_CHARS] + "\n\n[Additional context truncated for token limits]"

        return full_context


    # ─────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────
    # Node 3: Assemble context and generate final response
    # ─────────────────────────────────────────────────────────
    async def _generate_response(self, state: LangGraphState) -> dict:
        """
        Node 4: Compiles all retrieved documents + web search results into a
        single context string, calls LLM asynchronously for the final response,
        and appends the new Q&A turn to chat_history.

        Multi-turn awareness:
          - Reads the last CHAT_HISTORY_MAX_TURNS pairs from state["chat_history"].
          - Passes them to agenerate_response() so the LLM can give coherent
            follow-up answers (e.g. "As I mentioned earlier about Zakat...").
          - After generation, returns the new user+assistant pair in chat_history.
          - The _append_chat_history reducer concatenates it onto the stored list,
            so the checkpointer saves the growing history automatically.

        Returns a partial dict — only final_response and chat_history are updated.
        """
        error_message = state.get("error_message")
        retrieved_documents = state.get("retrieved_documents") or {}
        user_query = state.get("user_query", "")

        try:
            if error_message and not retrieved_documents:
                final_response = f"I apologize, but I encountered an error: {error_message}"
                return {"final_response": final_response}

            full_context = self._build_context(state)

            if not full_context.strip():
                final_response = (
                    "I could not find relevant information in the Islamic knowledge base "
                    "for your query. Please try rephrasing your question or consult a "
                    "qualified Islamic scholar."
                )
                return {"final_response": final_response}

            # Slice chat_history to the last MAX_TURNS pairs for the LLM context window
            max_turns = settings.CHAT_HISTORY_MAX_TURNS
            recent_history = (state.get("chat_history") or [])[-(max_turns * 2):]

            logger.info("Generating final response (history_turns=%d)...", len(recent_history) // 2)
            response = await self.openai_service.agenerate_response(
                user_query,
                full_context,
                chat_history=recent_history,
            )

            if response["status"] == "success":
                final_response = _remove_sources_section(response["message"])
                logger.info("Response generated successfully")
            else:
                final_response = (
                    f"I apologize, but I encountered an error while generating "
                    f"the response: {response['message']}"
                )
                logger.error(f"Response generation failed: {response['message']}")

        except Exception as e:
            logger.error(f"Error in _generate_response: {e}")
            final_response = f"I apologize, but I encountered an error: {str(e)}"

        # ── Append this turn to chat_history via the reducer ──────────
        # Returning chat_history here means _append_chat_history() will be called:
        #   new_history = _append_chat_history(current_checkpoint_history, new_turns)
        # The checkpointer then persists the merged list for the next turn.
        new_turn = [
            {"role": "user",      "content": user_query},
            {"role": "assistant", "content": final_response},
        ]

        return {
            "final_response": final_response,
            "chat_history": new_turn,   # reducer appends, does NOT overwrite
        }



    # ─────────────────────────────────────────────────────────
    # Helper: Web search via Tavily
    # ─────────────────────────────────────────────────────────
    @traceable(name="run_web_search", run_type="tool")
    def _run_web_search(self, query: str) -> list:
        """Run Tavily web search. Returns list of dicts with content/url/title."""
        try:
            if not self.tavily_client:
                logger.warning("Tavily client not initialized, skipping web search")
                return []

            search_results = self.tavily_client.search(
                query=f"{query} in Islam.",
                search_depth="basic",
                max_results=2,
                include_answer=False,
                include_raw_content=False,
                timeout=15,
            )

            web_documents = []
            for result in search_results.get("results", []):
                web_documents.append({
                    "content": (result.get("content", "") or "")[:800],
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                })

            return web_documents

        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return []



    # ─────────────────────────────────────────────────────────
    # Graph setup: called once from application.py lifespan
    # ─────────────────────────────────────────────────────────
    async def setup_graph(self):
        """
        Build and compile the LangGraph pipeline with the Redis checkpointer.

        Pipeline:
            rewrite_query → embed_query → classify_and_search → parallel_retrieve
                          → rerank_documents → generate_response → END

        Why async?
          AsyncRedisSaver uses an async Redis connection (aioredis under the hood).
          It must be initialised inside an async context, which is the FastAPI
          lifespan function — so we expose setup_graph() as a coroutine and call
          it from lifespan instead of from __init__.

        Checkpointer behaviour:
          - Keyed by thread_id (= session_id sent by the client).
          - After every node, the full state is checkpointed to Redis.
          - On the next call with the same thread_id, LangGraph restores the last
            checkpoint automatically before running the graph.
          - The _append_chat_history reducer on chat_history merges the new turn
            onto the restored history, so it grows across turns without any manual
            load/save logic in the application layer.
        """
        workflow = StateGraph(LangGraphState)

        # ── Register nodes ────────────────────────────────────────────
        workflow.add_node("rewrite_query",      _timed_node("rewrite", self._rewrite_query))
        workflow.add_node("embed_query",         _timed_node("embed", self._embed_query))
        workflow.add_node("classify_and_search", self._classify_and_search)
        workflow.add_node("parallel_retrieve",   _timed_node("retrieve", self._parallel_retrieve))
        workflow.add_node("rerank_documents",    _timed_node("rerank", self._rerank_documents))
        workflow.add_node("generate_response",   _timed_node("generate", self._generate_response))

        # ── Wire edges (linear flow) ──────────────────────────────────
        workflow.set_entry_point("rewrite_query")
        workflow.add_edge("rewrite_query",      "embed_query")
        workflow.add_edge("embed_query",         "classify_and_search")
        workflow.add_edge("classify_and_search", "parallel_retrieve")
        workflow.add_edge("parallel_retrieve",   "rerank_documents")
        workflow.add_edge("rerank_documents",    "generate_response")
        workflow.add_edge("generate_response",   END)

        # ── Attach checkpointer (AsyncRedisSaver with MemorySaver fallback) ─
        self._checkpointer_ctx = None
        try:
            self._checkpointer_ctx = AsyncRedisSaver.from_conn_string(settings.REDIS_URL)
            checkpointer = await self._checkpointer_ctx.__aenter__()
            # Ensure the Redis data structures LangGraph needs are ready
            await checkpointer.asetup()
            logger.info("LangGraph compiled with AsyncRedisSaver checkpointer (Redis: %s)", settings.REDIS_URL)
        except Exception as exc:
            logger.warning("Failed to initialize AsyncRedisSaver (%s). Falling back to MemorySaver.", exc)
            if self._checkpointer_ctx:
                try:
                    await self._checkpointer_ctx.__aexit__(None, None, None)
                except Exception:
                    pass
                self._checkpointer_ctx = None
            checkpointer = MemorySaver()
            logger.info("LangGraph compiled with in-memory MemorySaver checkpointer")

        self.graph = workflow.compile(checkpointer=checkpointer)


    async def teardown_graph(self):
        """Close the Redis checkpointer connection — called from lifespan teardown."""
        if getattr(self, "_checkpointer_ctx", None):
            await self._checkpointer_ctx.__aexit__(None, None, None)
            logger.info("AsyncRedisSaver checkpointer connection closed.")



    # ─────────────────────────────────────────────────────────
    # Public API: process a user query end-to-end
    # ─────────────────────────────────────────────────────────
    async def aquery(self, user_query: str, session_id: str = "default") -> str:
        """
        Process a user query asynchronously through the full compiled LangGraph.
        Returns the generated response string.

        session_id maps to LangGraph's thread_id — the checkpointer uses it to
        load the correct conversation checkpoint from Redis before running the
        graph, and saves the updated state (including new chat_history turn) after.
        """
        try:
            # Pass only the ephemeral fields for this turn.
            # The checkpointer automatically restores chat_history from the last
            # checkpoint for this thread_id before the graph starts running.
            initial_input = {
                "user_query": user_query,
                "standalone_query": user_query,  # Node 0 will overwrite if history exists
                "query_embedding": None,
                "stage_timings_ms": {},
                "web_search_results": [],
                "required_sources": [],
                "classification_reasoning": "",
                "retrieved_documents": {},
                "final_response": "",
                "error_message": None,
                "filters": None,
                # Do NOT pass chat_history here — the checkpointer injects it
            }

            config = {
                "run_name": "Islamic Chatbot Workflow",
                "configurable": {"thread_id": session_id},
            }

            logger.info("[aquery] session_id='%s' query='%s'", session_id, user_query[:80])
            final_state = await self.graph.ainvoke(initial_input, config=config)

            return final_state.get("final_response", "")

        except Exception as e:
            logger.error(f"Error in aquery pipeline: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"


    def query(self, user_query: str, session_id: str = "default") -> str:
        """
        Synchronous entry point — wraps aquery.
        """
        try:
            return asyncio.run(self.aquery(user_query, session_id))
        except RuntimeError:
            # Fallback if already inside a running loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return executor.submit(asyncio.run, self.aquery(user_query, session_id)).result()



    # ─────────────────────────────────────────────────────────
    # Public API: stream a user query via LangGraph native astream
    # ─────────────────────────────────────────────────────────
    async def query_stream(self, user_query: str, session_id: str = "default"):
        """
        Async generator that yields SSE event dicts using LangGraph's native astream().
        Executes the full compiled StateGraph:
            rewrite_query → embed_query → classify_and_search → parallel_retrieve
                          → rerank_documents → generate_response → END

        session_id is passed as thread_id in the LangGraph config so the checkpointer
        loads/saves the correct conversation history for this user session.

        - Mode 'updates': captures node completion events (progress updates).
        - Mode 'messages': captures token-by-token streaming directly from the LLM node.
        """
        try:
            logger.info("[STREAM] session_id='%s' query='%s'", session_id, user_query[:80])

            # Phase 1: Initial searching status
            yield {"status": "searching", "message": "Analyzing and retrieving from Islamic sources..."}

            initial_input = {
                "user_query": user_query,
                "standalone_query": user_query,
                "query_embedding": None,
                "stage_timings_ms": {},
                "web_search_results": [],
                "required_sources": [],
                "classification_reasoning": "",
                "retrieved_documents": {},
                "final_response": "",
                "error_message": None,
                "filters": None,
                # chat_history injected automatically by checkpointer
            }

            config = {
                "run_name": "Islamic Chatbot Workflow",
                "configurable": {"thread_id": session_id},
            }

            full_response = ""

            # Execute native LangGraph astream
            async for mode, payload in self.graph.astream(
                initial_input,
                stream_mode=["updates", "messages"],
                config=config,
            ):
                if mode == "updates":
                    # Transition status events
                    if "parallel_retrieve" in payload:
                        yield {"status": "generating", "message": "Composing response..."}
                    elif "generate_response" in payload:
                        node_state = payload["generate_response"]
                        if isinstance(node_state, dict):
                            full_response = node_state.get("final_response", full_response)

                elif mode == "messages":
                    msg, meta = payload
                    # Stream tokens emitted exclusively by the generate_response node
                    if meta.get("langgraph_node") == "generate_response" and msg.content:
                        full_response += msg.content
                        yield {"token": msg.content, "done": False}

            logger.info("[STREAM] Response streamed successfully via LangGraph native astream")
            yield {"done": True, "full_response": _remove_sources_section(full_response), "session_id": session_id}

        except Exception as e:
            logger.error(f"Error in query_stream: {e}")
            yield {"done": True, "full_response": f"I apologize, but I encountered an error: {str(e)}", "session_id": session_id}


# ─────────────────────────────────────────────────────────────────────────────
# Module-level graph export — required by LangSmith Studio
# ─────────────────────────────────────────────────────────────────────────────
# langgraph.json points to this symbol: "./services/langgraph_service.py:graph"
# LangSmith Studio imports it at startup to render the interactive visual graph.
#
# This is a *topology-only* compile (no checkpointer) so it can be built
# synchronously at import time without an event loop.  The real production graph
# (with AsyncRedisSaver) continues to live inside LangGraphService.setup_graph().
# ─────────────────────────────────────────────────────────────────────────────
def _build_graph_for_studio() -> object:
    """Return a compiled StateGraph suitable for LangSmith Studio inspection."""
    # Lazy import to avoid circular imports at module load time
    _svc = LangGraphService.__new__(LangGraphService)

    _wf = StateGraph(LangGraphState)
    _wf.add_node("rewrite_query",       _timed_node("rewrite", _svc._rewrite_query))
    _wf.add_node("embed_query",         _timed_node("embed", _svc._embed_query))
    _wf.add_node("classify_and_search", _svc._classify_and_search)
    _wf.add_node("parallel_retrieve",   _timed_node("retrieve", _svc._parallel_retrieve))
    _wf.add_node("rerank_documents",    _timed_node("rerank", _svc._rerank_documents))
    _wf.add_node("generate_response",   _timed_node("generate", _svc._generate_response))

    _wf.set_entry_point("rewrite_query")
    _wf.add_edge("rewrite_query",       "embed_query")
    _wf.add_edge("embed_query",          "classify_and_search")
    _wf.add_edge("classify_and_search", "parallel_retrieve")
    _wf.add_edge("parallel_retrieve",   "rerank_documents")
    _wf.add_edge("rerank_documents",    "generate_response")
    _wf.add_edge("generate_response",   END)

    return _wf.compile()          # no checkpointer — Studio only needs topology


graph = _build_graph_for_studio()