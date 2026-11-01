
import json
import numpy as np
from typing import Any, Optional, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from cachetools import TTLCache
import redis

from schemas.structured_outputs.query_classification import QueryClassificationSchema
from schemas.structured_outputs.query_rewrite import QueryRewriteSchema
from services.prompt_templates import QUERY_CLASSIFICATION_PROMPT, ENGLISH_RESPONSE_PROMPT, QUERY_REWRITE_PROMPT
from services.qdrant_service import QdrantService
from utils.custom_logger import setup_logger
from utils.config import settings

logger = setup_logger(__name__)


class OpenAIService:
    """Handles all OpenAI interactions: classification, embedding, and response generation."""

    def __init__(self, qdrant_service: Optional[QdrantService] = None):

        # Flagship LLM: Used for final response generation & streaming (gpt-4o) with Groq fallback
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            timeout=45,
            max_retries=2,
        )
        self.fallback_llm = ChatGroq(
            model=settings.GROQ_LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            timeout=45,
            max_retries=2,
        ) if settings.GROQ_API_KEY else None

        # Fast LLM: Used for classification, query rewrite, and structured extraction (gpt-4o-mini) with Groq fallback
        self.fast_llm = ChatOpenAI(
            model=settings.FAST_LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            timeout=20,
            max_retries=2,
        )
        self.fast_fallback_llm = ChatGroq(
            model=settings.FAST_GROQ_LLM_MODEL,
            api_key=settings.GROQ_API_KEY,
            timeout=20,
            max_retries=2,
        ) if settings.GROQ_API_KEY else None

        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.fallback_embeddings = None
 
        # In-memory fallback caches
        self._classification_cache = TTLCache(maxsize=500, ttl=3600)  
        self._semantic_cache: List[tuple] = []
        self._semantic_cache_max_size = 200
        self._semantic_similarity_threshold = 0.85

        # Initialize Redis connection (Layer 1)
        self.redis_client = None


        try:
            r = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD or None,
                decode_responses=True,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
                max_connections=50                                # added new connection pool
            )

            r.ping()
            self.redis_client = r       # The Python Redis client normally uses a connection pool internally.
            logger.info("Connected to Redis at %s:%d successfully.", settings.REDIS_HOST, settings.REDIS_PORT)

        except Exception as e:
            logger.warning("Redis not available (%s). Layer 1 will fallback to in-memory cache.", e)


        # Initialize Qdrant Service (Layer 2)
        try:
            if qdrant_service is None:
                self.qdrant_service = QdrantService()
            else:
                self.qdrant_service = qdrant_service
                
        except Exception as e:
            self.qdrant_service = None
            logger.warning("Qdrant service connection warning for caching: %s", e)



    def _get_from_exact_cache(self, normalized_query: str) -> Optional[QueryClassificationSchema]:
        """Layer 1 exact match: Checks Redis key-value first, then in-memory cache."""
        # 1. Try Redis
        if self.redis_client:
            try:
                cached_json = self.redis_client.get(f"exact_cache:{normalized_query}")
                if cached_json:
                    logger.info("Classification cache HIT (Layer 1 - Redis exact match)")
                    data = json.loads(cached_json)
                    return QueryClassificationSchema(**data)
            except Exception as e:
                logger.warning("Redis exact match error: %s", e)

        # 2. Fallback to in-memory TTLCache
        if normalized_query in self._classification_cache:
            logger.info("Classification cache HIT (Layer 1 - In-Memory exact match)")
            return self._classification_cache[normalized_query]

        return None

    def _get_from_semantic_cache(self, query_embedding: List[float]) -> Optional[dict]:
        """Layer 2 semantic match: Checks Qdrant vector collection first, then in-memory vectors."""
        # 1. Try Qdrant Cache Collection
        if self.qdrant_service:
            try:
                qdrant_hit = self.qdrant_service.search_cache_vector(
                    collection_name=settings.CLASSIFICATION_CACHE_COLLECTION_NAME,
                    query_vector=query_embedding,
                    score_threshold=self._semantic_similarity_threshold
                )
                if qdrant_hit:
                    logger.info("Classification cache HIT (Layer 2 - Qdrant vector match, sim=%.4f)", qdrant_hit["similarity_score"])
                    cls_obj = QueryClassificationSchema(**qdrant_hit["classification"])
                    return {
                        "classification": cls_obj,
                        "similarity_score": qdrant_hit["similarity_score"],
                        "matched_query": qdrant_hit["matched_query"]
                    }
            except Exception as e:
                logger.warning("Qdrant semantic cache search error: %s", e)

        # 2. Fallback to in-memory vector list
        if self._semantic_cache:
            best_similarity = 0.0
            best_result = None
            best_query = None
            for cached_emb, cached_result, cached_query in self._semantic_cache:
                similarity = self._cosine_similarity(query_embedding, cached_emb)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_result = cached_result
                    best_query = cached_query
            if best_similarity > self._semantic_similarity_threshold:
                logger.info("Classification cache HIT (Layer 2 - In-Memory vector match, sim=%.4f)", best_similarity)
                return {
                    "classification": best_result,
                    "similarity_score": round(best_similarity, 4),
                    "matched_query": best_query
                }

        return None

    def _write_to_caches(self, query: str, query_embedding: Optional[List[float]], classification: QueryClassificationSchema) -> None:
        """Writes classification result to Redis, Qdrant collection, and in-memory caches."""
        normalized = query.strip().lower()

        # Save to Memory
        self._classification_cache[normalized] = classification
        if query_embedding:
            self._semantic_cache.append((query_embedding, classification, normalized))
            if len(self._semantic_cache) > self._semantic_cache_max_size:
                self._semantic_cache.pop(0)

        # Save to Redis (Layer 1)
        if self.redis_client:
            try:
                json_str = classification.model_dump_json()
                if settings.REDIS_TTL > 0:
                    self.redis_client.setex(f"exact_cache:{normalized}", settings.REDIS_TTL, json_str)
                else:
                    self.redis_client.set(f"exact_cache:{normalized}", json_str)
                logger.info("Saved exact match to Redis for '%s'", normalized)
            except Exception as e:
                logger.warning("Failed to save exact cache to Redis: %s", e)

        # Save to Qdrant Cache Collection (Layer 2)
        if self.qdrant_service and query_embedding:
            try:
                self.qdrant_service.upsert_cache_vector(
                    collection_name=settings.CLASSIFICATION_CACHE_COLLECTION_NAME,
                    query_text=query,
                    query_vector=query_embedding,
                    classification_payload=classification.model_dump()
                )
            except Exception as e:
                logger.warning("Failed to save vector cache to Qdrant: %s", e)


    def classify_query(self, query: str, query_embedding: Optional[List[float]] = None) -> dict:
        """
        Classify a query to determine which Islamic sources to search.
        Uses 2-layer persistent caching (Redis + Qdrant) with in-memory fallbacks.
        """
        normalized = query.strip().lower()

        # Layer 1 Check
        l1_match = self._get_from_exact_cache(normalized)
        if l1_match:
            return {"status": "success", "message": l1_match}

        # Layer 2 Check
        if query_embedding:
            l2_match = self._get_from_semantic_cache(query_embedding)
            if l2_match:
                return {"status": "success", "message": l2_match["classification"]}

        # Layer 3: LLM Fallback Call
        logger.info("Classification cache MISS — calling LLM")
        result = self._process_request(QUERY_CLASSIFICATION_PROMPT, query, QueryClassificationSchema)

        if result["status"] == "success":
            classification = result["message"]
            self._write_to_caches(query, query_embedding, classification)

        return result


    def classify_query_with_metadata(self, query: str, query_embedding: Optional[List[float]] = None) -> dict:
        """
        Same as classify_query() but returns rich metadata for the Streamlit demo UI.
        """
        import time
        start = time.perf_counter()
        normalized = query.strip().lower()

        # Layer 1: Exact match
        l1_match = self._get_from_exact_cache(normalized)
        if l1_match:
            latency = (time.perf_counter() - start) * 1000
            return {
                "status": "success",
                "classification": l1_match,
                "cache_layer": "L1",
                "cache_label": "⚡ Exact Match Cache Hit (Redis/RAM)",
                "latency_ms": round(latency, 2),
                "similarity_score": None,
                "matched_query": normalized,
            }

        # Layer 2: Semantic similarity
        if query_embedding:
            l2_match = self._get_from_semantic_cache(query_embedding)
            if l2_match:
                latency = (time.perf_counter() - start) * 1000
                return {
                    "status": "success",
                    "classification": l2_match["classification"],
                    "cache_layer": "L2",
                    "cache_label": "🔵 Semantic Cache Hit (Qdrant/RAM)",
                    "latency_ms": round(latency, 2),
                    "similarity_score": l2_match["similarity_score"],
                    "matched_query": l2_match["matched_query"],
                }

        # Layer 3: LLM fallback
        result = self._process_request(QUERY_CLASSIFICATION_PROMPT, query, QueryClassificationSchema)
        latency = (time.perf_counter() - start) * 1000

        if result["status"] == "success":
            classification = result["message"]
            self._write_to_caches(query, query_embedding, classification)
            return {
                "status": "success",
                "classification": classification,
                "cache_layer": "LLM",
                "cache_label": "🤖 LLM Fallback",
                "latency_ms": round(latency, 2),
                "similarity_score": None,
                "matched_query": None,
            }

        return {"status": "error", "message": result["message"]}


    def _get_or_init_fallback_embeddings(self) -> HuggingFaceEmbeddings:
        """Lazily initialise the HuggingFace fallback embedding model."""
        if not self.fallback_embeddings:
            hf_kwargs = {"trust_remote_code": True}
            if settings.HF_TOKEN:
                hf_kwargs["token"] = settings.HF_TOKEN
            self.fallback_embeddings = HuggingFaceEmbeddings(
                model_name=settings.FALLBACK_EMBEDDING_MODEL,
                model_kwargs=hf_kwargs,
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("HuggingFace fallback embeddings initialised (%s).", settings.FALLBACK_EMBEDDING_MODEL)
        return self.fallback_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a query string.

        Routing is controlled by settings.USE_PRIMARY_EMBEDDING:
          True  → try OpenAI first; fall back to HuggingFace on error.
          False → go directly to HuggingFace (OpenAI is skipped entirely).
        """
        if not settings.USE_PRIMARY_EMBEDDING:
            logger.info("Primary embedding is DISABLED — using HuggingFace (%s) directly.", settings.FALLBACK_EMBEDDING_MODEL)
            return self._get_or_init_fallback_embeddings().embed_query(query)

        try:
            return self.embeddings.embed_query(query)
        except Exception as e:
            logger.warning(
                "Primary OpenAI embedding failed (%s). Falling back to HuggingFace (%s)...",
                e, settings.FALLBACK_EMBEDDING_MODEL
            )
            try:
                return self._get_or_init_fallback_embeddings().embed_query(query)
            except Exception as fallback_err:
                logger.error("Fallback embedding also failed: %s", fallback_err)
                raise fallback_err
 
 
    def rewrite_query(self, user_query: str, chat_history: List[dict]) -> str:
        """
        Rewrites a potentially ambiguous follow-up question into a fully
        standalone search query by using the recent conversation history.

        Called synchronously inside the rewrite_query node which runs in the
        classification_executor ThreadPoolExecutor.

        How it works:
          1. Formats conversation history into a structured prompt context block
             (avoiding alternating Human/AI messages which trigger chat assistant replies).
          2. Invokes LLM with structured output (QueryRewriteSchema) to strictly guarantee
             a standalone query string.
          3. Validates against answering prefixes and falls back gracefully.
        """
        if not chat_history:
            # No history → nothing to resolve, return as-is
            return user_query

        try:
            # Build conversation history summary block
            history_lines = []
            for turn in chat_history:
                role = "User" if turn.get("role") == "user" else "Assistant"
                content = str(turn.get("content", "")).strip()
                # Truncate lengthy assistant turns to keep context focused on query subjects
                if role == "Assistant" and len(content) > 250:
                    content = content[:250] + "..."
                if content:
                    history_lines.append(f"{role}: {content}")

            formatted_history = "\n".join(history_lines)

            task_input = (
                f"Conversation History:\n{formatted_history}\n\n"
                f"User Follow-Up Query:\n\"{user_query}\"\n\n"
                f"Task: Rewrite the follow-up query into a single standalone search question. "
                f"DO NOT answer the question. Only output the standalone query."
            )

            messages = [
                SystemMessage(content=QUERY_REWRITE_PROMPT),
                HumanMessage(content=task_input),
            ]

            # Use fast structured LLM for ultra-fast query reformulations.
            # Routing: if USE_PRIMARY_LLM is False skip OpenAI and go straight to Groq.
            if settings.USE_PRIMARY_LLM:
                structured_llm = self.fast_llm.with_structured_output(QueryRewriteSchema)
            else:
                if self.fast_fallback_llm is None:
                    logger.warning("USE_PRIMARY_LLM=False but no fallback LLM available — using primary.")
                    structured_llm = self.fast_llm.with_structured_output(QueryRewriteSchema)
                else:
                    logger.info("USE_PRIMARY_LLM=False — routing rewrite_query to Groq (%s).", settings.FAST_GROQ_LLM_MODEL)
                    structured_llm = self.fast_fallback_llm.with_structured_output(QueryRewriteSchema)
            res = structured_llm.invoke(messages)

            rewritten = ""
            if isinstance(res, QueryRewriteSchema):
                rewritten = res.standalone_query.strip()
            elif isinstance(res, dict):
                rewritten = str(res.get("standalone_query", "")).strip()
            elif hasattr(res, "content"):
                rewritten = str(res.content).strip()

            # Clean outer quotes if any
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1].strip()

            # Guardrail: Check if the model answered instead of rewriting
            answering_prefixes = [
                "please note", "in islam", "according to", "scholars hold",
                "the quran and hadith", "there is no explicit", "it is important to note",
                "as mentioned", "based on"
            ]
            lower_rewritten = rewritten.lower()
            if any(lower_rewritten.startswith(prefix) for prefix in answering_prefixes):
                logger.warning(
                    "Query rewrite attempted to answer ('%s') — falling back to context-merged query.",
                    rewritten[:60]
                )
                # Fall back to pairing last user query with follow-up
                last_user_query = ""
                for turn in reversed(chat_history):
                    if turn.get("role") == "user":
                        last_user_query = turn.get("content", "").strip()
                        break
                rewritten = f"{last_user_query} - {user_query}" if last_user_query else user_query

            logger.info("Query rewritten: '%s' → '%s'", user_query[:60], rewritten[:60])
            return rewritten if rewritten else user_query

        except Exception as e:
            logger.warning("Query rewrite failed (%s) — using original query.", e)
            return user_query


    def generate_response(self, query: str, context: str) -> dict:
        """Generate the final comprehensive Islamic response using retrieved context."""
        prompt = ENGLISH_RESPONSE_PROMPT.replace("{context}", context)
        return self._process_request(prompt, query, schema=None)


    async def agenerate_response(self, query: str, context: str, chat_history: List[dict] = None) -> dict:
        """
        Generate the final comprehensive Islamic response asynchronously using ainvoke.
        Enables LangGraph astream(stream_mode='messages') to stream tokens natively.

        Multi-turn awareness:
          - chat_history contains the last N Q&A turns as {role, content} dicts.
          - We build a proper LangChain message list so the LLM sees the full
            conversation context, enabling coherent follow-up answers.
          - The system prompt (with retrieved context) always comes first.
          - History turns are interleaved as HumanMessage / AIMessage pairs.
          - The current user query is always the final HumanMessage.
        """
        prompt = ENGLISH_RESPONSE_PROMPT.replace("{context}", context)
        try:
            # Start with the system prompt that includes retrieved context
            messages = [SystemMessage(content=prompt)]

            # Inject prior conversation turns so the LLM can answer coherently
            if chat_history:
                for turn in chat_history:
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    else:
                        messages.append(AIMessage(content=content))

            # Current user question is always the final message
            messages.append(HumanMessage(content=query))

            def _trim_for_groq(msgs: list) -> list:
                """Truncate the system message to stay within Groq TPM limits."""
                out = list(msgs)
                if out and len(out[0].content) > 12000:
                    out[0] = SystemMessage(content=out[0].content[:12000] + "\n\n[Context truncated for Groq TPM limit]")
                return out

            if not settings.USE_PRIMARY_LLM:
                # Primary LLM closed → route directly to Groq, no OpenAI attempt
                if self.fallback_llm:
                    logger.info("USE_PRIMARY_LLM=False — routing agenerate_response to Groq (%s).", settings.GROQ_LLM_MODEL)
                    response = await self.fallback_llm.ainvoke(_trim_for_groq(messages))
                else:
                    raise RuntimeError("USE_PRIMARY_LLM=False but no fallback_llm configured.")
            else:
                try:
                    response = await self.llm.ainvoke(messages)
                except Exception as openai_err:
                    if self.fallback_llm:
                        logger.warning("OpenAI ainvoke failed (%s). Falling back to Groq (%s)...", openai_err, settings.GROQ_LLM_MODEL)
                        response = await self.fallback_llm.ainvoke(_trim_for_groq(messages))
                    else:
                        raise openai_err

            return {
                "status": "success",
                "message": response.content,
            }
        except Exception as e:
            logger.error(f"Error in agenerate_response: {e}")
            return {"status": "error", "message": str(e)}


    async def generate_response_stream(self, query: str, context: str):
        """
        Async generator that streams the LLM response token-by-token.
        Uses ChatOpenAI.astream() (with Groq fallback) which yields AIMessageChunk objects.
        Each yielded value is a string token fragment.
        """
        prompt = ENGLISH_RESPONSE_PROMPT.replace("{context}", context)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=query),
        ]

        def _trim(msgs: list) -> list:
            out = list(msgs)
            if out and len(out[0].content) > 12000:
                out[0] = SystemMessage(content=out[0].content[:12000] + "\n\n[Context truncated for Groq TPM limit]")
            return out

        try:
            if not settings.USE_PRIMARY_LLM:
                # Primary LLM closed → stream directly from Groq
                if self.fallback_llm:
                    logger.info("USE_PRIMARY_LLM=False — streaming via Groq (%s).", settings.GROQ_LLM_MODEL)
                    async for chunk in self.fallback_llm.astream(_trim(messages)):
                        if chunk.content:
                            yield chunk.content
                else:
                    raise RuntimeError("USE_PRIMARY_LLM=False but no fallback_llm configured.")
            else:
                try:
                    async for chunk in self.llm.astream(messages):
                        if chunk.content:
                            yield chunk.content
                except Exception as openai_err:
                    if self.fallback_llm:
                        logger.warning("OpenAI astream failed (%s). Falling back to Groq stream...", openai_err)
                        async for chunk in self.fallback_llm.astream(_trim(messages)):
                            if chunk.content:
                                yield chunk.content
                    else:
                        raise openai_err
        except Exception as e:
            logger.error(f"Error in generate_response_stream: {e}")
            yield f"\n\n[Error: {str(e)}]"


    def _process_request(self, prompt: str, text: str, schema=None, use_fast_llm: bool = False) -> dict:
        """Generic LLM call with primary / fallback routing.

        Routing (controlled by settings.USE_PRIMARY_LLM):
          True  → try primary (OpenAI) first; on error fall back to Groq.
          False → skip OpenAI entirely and call Groq directly.
        """
        try:
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=text)
            ]

            # Select the right model pair based on task type
            primary_llm   = self.fast_llm         if (schema or use_fast_llm) else self.llm
            fallback_model = self.fast_fallback_llm if (schema or use_fast_llm) else self.fallback_llm
            model_label    = "fast" if (schema or use_fast_llm) else "flagship"

            if not settings.USE_PRIMARY_LLM:
                # Primary LLM is CLOSED → go straight to Groq
                if fallback_model:
                    logger.info(
                        "USE_PRIMARY_LLM=False — routing _process_request (%s) directly to Groq.", model_label
                    )
                    fallback_instance = fallback_model.with_structured_output(schema) if schema else fallback_model
                    response = fallback_instance.invoke(messages)
                else:
                    raise RuntimeError(
                        f"USE_PRIMARY_LLM=False but no fallback model available for '{model_label}' requests."
                    )
            else:
                try:
                    llm_instance = primary_llm.with_structured_output(schema) if schema else primary_llm
                    response = llm_instance.invoke(messages)
                except Exception as primary_err:
                    if fallback_model:
                        logger.warning("Primary LLM request failed (%s). Falling back to Groq...", primary_err)
                        fallback_instance = fallback_model.with_structured_output(schema) if schema else fallback_model
                        response = fallback_instance.invoke(messages)
                    else:
                        raise primary_err

            return {
                "status": "success",
                "message": response.content if not schema else response
            }

        except Exception as e:
            logger.error(f"Error in _process_request: {e}")
            return {"status": "error", "message": f"Error processing request: {e}"}


    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)


