import sys
from pathlib import Path
from typing import List, Any, Optional, Dict

import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    Prefetch,
    SparseVector,
    Fusion,
    FusionQuery,
)
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.custom_logger import setup_logger
from utils.config import settings


from qdrant_client.http import models


logger = setup_logger(__name__)


class QdrantService:

    def __init__(
        self,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):

        self.qdrant_url = qdrant_url or settings.QDRANT_URL
        self.qdrant_api_key = qdrant_api_key or settings.QDRANT_API_KEY
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL

        logger.info("Connecting to Qdrant at %s ...", self.qdrant_url)

        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key or None,
            timeout=10,
        )
    
        logger.info("Connected to Qdrant successfully.")
        self.ensure_cache_collection()

    def ensure_cache_collection(self) -> None:
        """Ensures the classification_cache collection exists in Qdrant."""
        try:
            self.setup_collection(
                collection_name=settings.CLASSIFICATION_CACHE_COLLECTION_NAME,
                embedding_dimension=settings.EMBEDDING_DIMENSION,
                force_recreate=False
            )
        except Exception as e:
            logger.warning("Failed to auto-setup Qdrant cache collection: %s", e)


    def setup_collection(
        self,
        collection_name: str,
        embedding_dimension: Optional[int] = None,
        force_recreate: Optional[bool] = None,
        with_sparse_vectors: bool = True,
    ) -> None:
        """
        Creates or ensures a Qdrant collection exists. Resets if force_recreate is True.
        When with_sparse_vectors=True (and the collection is not the cache collection),
        the collection is created with both a dense vector config AND a sparse vector config
        for BM25 hybrid search. The sparse vector name is taken from settings.SPARSE_VECTOR_NAME.
        """
        dimension = embedding_dimension or settings.EMBEDDING_DIMENSION
        should_recreate = settings.FORCE_RECREATE if force_recreate is None else force_recreate

        # The classification cache only needs dense vectors (no hybrid search needed there)
        is_cache_collection = (collection_name == settings.CLASSIFICATION_CACHE_COLLECTION_NAME)
        enable_sparse = with_sparse_vectors and not is_cache_collection and settings.HYBRID_SEARCH_ENABLED

        existing_collections = [col.name for col in self.client.get_collections().collections]

        if should_recreate and collection_name in existing_collections:
            logger.warning("FORCE_RECREATE=True. Deleting collection '%s' for clean re-index.", collection_name)
            self.client.delete_collection(collection_name=collection_name)
            existing_collections.remove(collection_name)

        if collection_name not in existing_collections:
            logger.info(
                "Creating fresh Qdrant collection: '%s' (sparse=%s) ...",
                collection_name,
                enable_sparse,
            )

            if enable_sparse:
                # Hybrid collection: named dense vector + named sparse vector
                # Dense vector is named "dense"; sparse vector uses settings.SPARSE_VECTOR_NAME
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=dimension,
                            distance=Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        settings.SPARSE_VECTOR_NAME: SparseVectorParams(
                            index=SparseIndexParams(
                                on_disk=False,  # keep in RAM for fast retrieval
                            )
                        )
                    },
                )
            else:
                # Dense-only collection (cache or hybrid disabled)
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE,
                    ),
                )

            logger.info(
                "Collection '%s' created (dim=%d, COSINE, sparse=%s)",
                collection_name, dimension, enable_sparse,
            )
            if hasattr(self, "_dense_vector_collections"):
                self._dense_vector_collections.pop(collection_name, None)
            if hasattr(self, "_hybrid_supported_collections"):
                self._hybrid_supported_collections.pop(collection_name, None)

        else:
            info = self.client.get_collection(collection_name)
            logger.info("Collection '%s' exists with %d existing point(s).", collection_name, info.points_count)


    def embed_and_upload(
        self,
        chunks: List[Document],
        collection_name: str,
        embeddings: Any,
        batch_size: Optional[int] = None,
        force_recreate: Optional[bool] = None,
        vector_name: Optional[str] = None,
    ) -> None:
        """
        Embeds document chunks in batches and auto-resumes from last uploaded point if interrupted.

        Args:
            vector_name: Optional named vector field to use. Pass "dense" when the collection
                         was created with named vectors (hybrid schema). Leave None for legacy
                         dense-only collections that use the default unnamed vector.
        """
        if not chunks:
            logger.warning("No chunks to upload for '%s', skipping.", collection_name)
            return

        batch_sz = batch_size or settings.BATCH_SIZE
        should_recreate = settings.FORCE_RECREATE if force_recreate is None else force_recreate

        total_chunks = len(chunks)
        info = self.client.get_collection(collection_name)
        existing_count = info.points_count

        if existing_count >= total_chunks and not should_recreate:
            logger.info("Collection '%s' is already fully uploaded (%d / %d chunks). Skipping!", collection_name, existing_count, total_chunks)
            return

        start_idx = existing_count if not should_recreate else 0

        if start_idx > 0:
            logger.info("🔄 Resuming '%s': Skipping first %d chunk(s). Uploading remaining %d chunk(s) ...",
                        collection_name, start_idx, total_chunks - start_idx)
        else:
            logger.info("🚀 Starting upload of %d chunk(s) to '%s' ...", total_chunks, collection_name)

        total_batches = (total_chunks - start_idx + batch_sz - 1) // batch_sz

        for i in range(start_idx, total_chunks, batch_sz):
            batch = chunks[i : i + batch_sz]
            current_batch_num = (i - start_idx) // batch_sz + 1

            logger.info(
                "Uploading Batch %d/%d (chunks %d–%d / %d) to '%s' ...",
                current_batch_num, total_batches, i + 1, min(i + batch_sz, total_chunks), total_chunks, collection_name
            )

            kwargs = dict(
                documents=batch,
                embedding=embeddings,
                url=self.qdrant_url,
                api_key=self.qdrant_api_key or None,
                collection_name=collection_name,
                force_recreate=False,
            )
            # When collection uses named dense vector (hybrid schema), we must tell
            # QdrantVectorStore which vector field to write into.
            if vector_name:
                kwargs["vector_name"] = vector_name

            QdrantVectorStore.from_documents(**kwargs)

        logger.info("✅ Upload complete for '%s' (Total points in Qdrant: %d)", collection_name, self.client.get_collection(collection_name).points_count)


    def _collection_has_dense_vector(self, collection_name: str) -> bool:
        """Check if the collection was configured with a named 'dense' vector."""
        try:
            if not hasattr(self, "_dense_vector_collections"):
                self._dense_vector_collections = {}
            if collection_name in self._dense_vector_collections:
                return self._dense_vector_collections[collection_name]
            info = self.client.get_collection(collection_name)
            vectors_config = info.config.params.vectors
            has_dense = isinstance(vectors_config, dict) and "dense" in vectors_config
            self._dense_vector_collections[collection_name] = has_dense
            return has_dense
        except Exception:
            return False

    def collection_supports_hybrid(self, collection_name: str) -> bool:
        """Check if the collection was configured with both named 'dense' and 'sparse' vectors."""
        try:
            if not hasattr(self, "_hybrid_supported_collections"):
                self._hybrid_supported_collections = {}
            if collection_name in self._hybrid_supported_collections:
                return self._hybrid_supported_collections[collection_name]
            info = self.client.get_collection(collection_name)
            vectors_config = info.config.params.vectors
            sparse_config = info.config.params.sparse_vectors
            has_dense = isinstance(vectors_config, dict) and "dense" in vectors_config
            has_sparse = isinstance(sparse_config, dict) and (settings.SPARSE_VECTOR_NAME in sparse_config or len(sparse_config) > 0)
            is_hybrid = bool(has_dense and has_sparse)
            self._hybrid_supported_collections[collection_name] = is_hybrid
            return is_hybrid
        except Exception:
            return False


    def get_vector_store(self, collection_name: str, embeddings: Any) -> QdrantVectorStore:
        """Returns a LangChain QdrantVectorStore instance for querying an existing collection."""
        kwargs: dict = {
            "client": self.client,
            "collection_name": collection_name,
            "embedding": embeddings,
        }
        if collection_name != settings.CLASSIFICATION_CACHE_COLLECTION_NAME and self._collection_has_dense_vector(collection_name):
            kwargs["vector_name"] = "dense"
        return QdrantVectorStore(**kwargs)


    def get_retriever(self, collection_name: str, embeddings: Any, k: int = 4, search_type: str = "similarity"):
        """Returns a LangChain Retriever instance for RAG pipelines."""
        vector_store = self.get_vector_store(collection_name, embeddings)
        return vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})


    def search(self, collection_name: str, query: str, embeddings: Any, limit: int = 4) -> List[Document]:
        """Performs similarity search against a specific collection."""
        vector_store = self.get_vector_store(collection_name, embeddings)
        return vector_store.similarity_search(query=query, k=limit)


    def search_by_vector(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
        query_filter: Optional[models.Filter] = None,
    ) -> List[dict]:
        """
        Dense-only (semantic) search using a pre-computed embedding vector.
        Used for the classification cache collection and as fallback when hybrid is disabled.
        Returns raw dicts with 'content', 'metadata', and 'score' keys.
        """
        try:
            using_vector = "dense" if (collection_name != settings.CLASSIFICATION_CACHE_COLLECTION_NAME and self._collection_has_dense_vector(collection_name)) else None
            query_kwargs: dict = {
                "collection_name": collection_name,
                "limit": limit,
                "query_filter": query_filter,
                "with_payload": True,
                "with_vectors": False,
            }
            if using_vector:
                query_kwargs["using"] = using_vector

            if hasattr(self.client, "query_points"):
                try:
                    res = self.client.query_points(
                        query=query_vector,
                        **query_kwargs,
                    )
                except Exception as e:
                    if ("Not existing vector name error" in str(e) or "vector name" in str(e).lower()) and "using" in query_kwargs:
                        logger.warning("Vector name '%s' not found in collection '%s'. Retrying unnamed vector.", using_vector, collection_name)
                        query_kwargs.pop("using", None)
                        res = self.client.query_points(query=query_vector, **query_kwargs)
                    else:
                        raise
                results = res.points
            else:
                if using_vector:
                    query_kwargs.pop("using", None)
                    query_kwargs["vector_name"] = using_vector
                try:
                    results = self.client.search(
                        query_vector=query_vector,
                        **query_kwargs,
                    )
                except Exception as e: 
                    if ("Not existing vector name error" in str(e) or "vector name" in str(e).lower()) and "vector_name" in query_kwargs:
                        logger.warning("Vector name '%s' not found in collection '%s'. Retrying unnamed vector.", using_vector, collection_name)
                        query_kwargs.pop("vector_name", None)
                        results = self.client.search(query_vector=query_vector, **query_kwargs)
                    else:
                        raise

            return [
                {
                    "content": r.payload.get("page_content", ""),
                    "metadata": r.payload.get("metadata", {}),
                    "score": r.score,
                }
                for r in results
            ]
        except Exception as e:
            logger.error("Error in search_by_vector for '%s': %s", collection_name, e)
            return []


    def hybrid_search_by_vector(
        self,
        collection_name: str,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        limit: int = 5,
        query_filter: Optional[models.Filter] = None,
    ) -> List[dict]:
        """
        Hybrid search using RRF fusion of dense (semantic) + sparse (BM25 keyword) vectors.

        How it works:
          1. Prefetch top-K candidates from the dense vector index.
          2. Prefetch top-K candidates from the sparse (BM25) vector index.
          3. Qdrant's native RRF (Reciprocal Rank Fusion) merges both ranked lists
             into a single re-ranked result list.

        Args:
            collection_name:  Target Qdrant collection (must have both dense + sparse configs).
            dense_vector:     Pre-computed OpenAI dense embedding (list of floats, dim=1536).
            sparse_indices:   BM25 sparse vector token indices (from fastembed).
            sparse_values:    BM25 sparse vector token weights (from fastembed).
            limit:            Number of final results to return after RRF fusion.
            query_filter:     Optional Qdrant metadata filter (surah_number, ayah_number, etc.).

        Returns:
            List of dicts with 'content', 'metadata', and 'score' keys.
        """
        try:
            if not self.collection_supports_hybrid(collection_name):
                logger.debug(
                    "Collection '%s' does not have hybrid schema (named 'dense' and '%s' sparse). Using dense search.",
                    collection_name, settings.SPARSE_VECTOR_NAME,
                )
                return self.search_by_vector(collection_name, dense_vector, limit, query_filter)

            # Prefetch a larger candidate pool from each index, then fuse down to `limit`
            prefetch_limit = limit * 3

            prefetch_dense = Prefetch(
                query=dense_vector,
                using="dense",           # name matching vectors_config key in setup_collection
                limit=prefetch_limit,
                filter=query_filter,
            )

            prefetch_sparse = Prefetch(
                query=SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
                using=settings.SPARSE_VECTOR_NAME,   # matches sparse_vectors_config key
                limit=prefetch_limit,
                filter=query_filter,
            )

            res = self.client.query_points(
                collection_name=collection_name,
                prefetch=[prefetch_dense, prefetch_sparse],
                query=FusionQuery(fusion=Fusion.RRF),   # native RRF re-ranking
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            return [
                {
                    "content": r.payload.get("page_content", ""),
                    "metadata": r.payload.get("metadata", {}),
                    "score": r.score,
                }
                for r in res.points
            ]

        except Exception as e:
            logger.error(
                "Error in hybrid_search_by_vector for '%s': %s. Falling back to dense-only.",
                collection_name, e,
            )
            # Graceful fallback to dense-only search if hybrid fails
            return self.search_by_vector(collection_name, dense_vector, limit, query_filter)


    def upsert_cache_vector(
        self,
        collection_name: str,
        query_text: str,
        query_vector: List[float],
        classification_payload: dict,
    ) -> None:
        """Upserts a classification query vector and payload metadata into Qdrant cache collection."""
        try:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, query_text.strip().lower()))
            point = PointStruct(
                id=point_id,
                vector=query_vector,
                payload={
                    "query_text": query_text,
                    "classification": classification_payload,
                }
            )
            self.client.upsert(
                collection_name=collection_name,
                points=[point]
            )
            logger.info("Upserted classification vector for '%s' into Qdrant collection '%s'", query_text, collection_name)
        except Exception as e:
            logger.error("Error upserting vector into cache collection '%s': %s", collection_name, e)


    def search_cache_vector(
        self,
        collection_name: str,
        query_vector: List[float],
        score_threshold: float = 0.85
    ) -> Optional[dict]:
        """
        Searches the Qdrant classification cache collection using query vector.
        Returns closest match dict if score exceeds score_threshold, else None.
        """
        try:
            if hasattr(self.client, "query_points"):
                res = self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=1,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False
                )
                results = res.points
            else:
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=1,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False
                )

            if results:
                best_match = results[0]
                return {
                    "matched_query": best_match.payload.get("query_text", ""),
                    "classification": best_match.payload.get("classification", {}),
                    "similarity_score": round(float(best_match.score), 4),
                }
            return None
            
        except Exception as e:
            logger.error("Error searching cache collection '%s': %s", collection_name, e)
            return None



    def build_qdrant_filter(self, collection_name: str, filters: Optional[dict]) -> Optional[models.Filter]:
        """
        Builds a collection-specific Qdrant models.Filter based on the extracted metadata filters.
        Maps fields to each collection's exact payload schema:
          - Quran: metadata.reference ("2:153") or metadata.ayah_number (153)
          - Tafsir: metadata.surah_number (2) and metadata.ayah_number (153)
          - Hadith: metadata.title ("Sahih al-Bukhari")
        """
        if not filters:
            return None

        conditions = []
        c_name = collection_name.lower()

        # ── 1. Quran Collection ─────────────────────────────────────────────
        if "quran" in c_name:
            surah_num = filters.get("surah_number")
            ayah_num = filters.get("ayah_number") if filters.get("ayah_number") is not None else filters.get("verse_number")

            if surah_num is not None and ayah_num is not None:
                ref_str = f"{surah_num}:{ayah_num}"
                conditions.append(
                    models.FieldCondition(
                        key="metadata.reference",
                        match=models.MatchValue(value=ref_str)
                    )
                )
            elif ayah_num is not None:
                conditions.append(
                    models.FieldCondition(
                        key="metadata.ayah_number",
                        match=models.MatchValue(value=int(ayah_num))
                    )
                )

        # ── 2. Tafsir Collection ────────────────────────────────────────────
        elif "tafsir" in c_name or "tafseer" in c_name:
            surah_num = filters.get("surah_number")
            ayah_num = filters.get("ayah_number") if filters.get("ayah_number") is not None else filters.get("verse_number")

            if surah_num is not None:
                conditions.append(
                    models.FieldCondition(
                        key="metadata.surah_number",
                        match=models.MatchValue(value=int(surah_num))
                    )
                )
            if ayah_num is not None:
                conditions.append(
                    models.FieldCondition(
                        key="metadata.ayah_number",
                        match=models.MatchValue(value=int(ayah_num))
                    )
                )

        # ── 3. Hadith Collection ───────────────────────────────────────────
        elif "hadith" in c_name:
            book = filters.get("hadith_book") or filters.get("hadith_book_name")
            if book:
                conditions.append(
                    models.FieldCondition(
                        key="metadata.title",
                        match=models.MatchText(text=str(book))
                    )
                )

        if conditions:
            return models.Filter(must=conditions)

        return None
