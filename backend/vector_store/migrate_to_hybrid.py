"""
Zero-Cost Hybrid Migration Script for Qdrant Collections.
Converts legacy dense-only collections into dual-vector (named 'dense' + BM25 'sparse') collections
WITHOUT re-calling OpenAI embeddings API.

How it works:
  1. Scrolls existing points (id, dense vector, and full payload metadata) directly from Qdrant.
  2. Computes BM25 sparse vectors locally via fastembed Bm25 model (CPU, 0 API calls).
  3. Inserts into a temporary staging collection: <name>_hybrid_staging with both vectors.
  4. Once 100% verified against original points count, safely recreates the original collection
     with the hybrid schema and moves the points over.
  5. Cleans up staging collections.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root and vector_store folder to sys.path
vector_store_dir = Path(__file__).resolve().parent
project_root = vector_store_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(vector_store_dir) not in sys.path:
    sys.path.insert(0, str(vector_store_dir))

from qdrant_client.models import (
    PointStruct,
    SparseVector,
)
from fastembed.sparse.bm25 import Bm25

from utils.custom_logger import setup_logger
from utils.config import settings
from services.qdrant_service import QdrantService

logger = setup_logger(__name__)


def migrate_collection_to_hybrid(
    qdrant_service: QdrantService,
    bm25_model: Bm25,
    collection_name: str,
    batch_size: int = 500,
) -> bool:
    """
    Migrates a single collection to the new named dense + sparse hybrid schema.
    Returns True if successfully migrated, False if skipped or failed.
    """
    client = qdrant_service.client
    existing_collections = [c.name for c in client.get_collections().collections]

    if collection_name not in existing_collections:
        logger.warning("Collection '%s' does not exist in Qdrant. Skipping.", collection_name)
        return False

    # Check if already hybrid
    if qdrant_service.collection_supports_hybrid(collection_name):
        logger.info(
            "Collection '%s' already has hybrid schema ('dense' + '%s'). Skipping!",
            collection_name, settings.SPARSE_VECTOR_NAME
        )
        return False

    info = client.get_collection(collection_name)
    total_points = info.points_count
    if total_points == 0:
        logger.warning("Collection '%s' has 0 points. Skipping.", collection_name)
        return False

    logger.info("=" * 60)
    logger.info("MIGRATING '%s' TO HYBRID SCHEMA (Total points: %d)", collection_name.upper(), total_points)
    logger.info("=" * 60)

    staging_col = f"{collection_name}_hybrid_staging"

    # Step 1: Create staging collection with hybrid schema
    logger.info("Step 1/4: Creating hybrid staging collection '%s' ...", staging_col)
    qdrant_service.setup_collection(
        collection_name=staging_col,
        embedding_dimension=settings.EMBEDDING_DIMENSION,
        force_recreate=True,
        with_sparse_vectors=True,
    )

    # Step 2: Scroll original points, compute BM25 locally, upload to staging
    logger.info("Step 2/4: Scrolling points and computing local BM25 sparse vectors ...")
    offset = None
    migrated_count = 0
    start_time = time.time()

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not records:
            break

        # Extract text for BM25
        texts = [r.payload.get("page_content", "") if r.payload else "" for r in records]

        # Compute BM25 sparse embeddings (fast generator -> list)
        sparse_gen = bm25_model.embed(texts, batch_size=len(texts))
        sparse_embeddings = list(sparse_gen)

        # Assemble new points with named dense + sparse vectors
        points_to_upsert: List[PointStruct] = []
        for r, sparse_emb in zip(records, sparse_embeddings):
            dense_vector = r.vector if isinstance(r.vector, list) else r.vector.get("dense")
            points_to_upsert.append(
                PointStruct(
                    id=r.id,
                    vector={
                        "dense": dense_vector,
                        settings.SPARSE_VECTOR_NAME: SparseVector(
                            indices=sparse_emb.indices.tolist(),
                            values=sparse_emb.values.tolist(),
                        ),
                    },
                    payload=r.payload or {},
                )
            )

        client.upsert(
            collection_name=staging_col,
            points=points_to_upsert,
            wait=True,
        )

        migrated_count += len(records)
        elapsed = time.time() - start_time
        rate = migrated_count / elapsed if elapsed > 0 else 0
        pct = (migrated_count / total_points) * 100
        logger.info(
            "[%s Staging] %d / %d points processed (%.1f%%) — %.0f points/sec",
            collection_name, migrated_count, total_points, pct, rate
        )

        offset = next_offset
        if offset is None:
            break

    # Step 3: Verify staging points count
    staging_info = client.get_collection(staging_col)
    if staging_info.points_count != total_points:
        logger.error(
            "Validation failed! Staging count (%d) != Original count (%d). Aborting migration for '%s'!",
            staging_info.points_count, total_points, collection_name
        )
        return False

    logger.info(
        "✅ Verification passed: all %d points staged with dense + sparse vectors.",
        staging_info.points_count
    )

    # Step 4: Safe swap: Delete old collection, recreate with hybrid schema, transfer from staging
    logger.info("Step 3/4: Safely recreating '%s' with hybrid schema ...", collection_name)
    client.delete_collection(collection_name=collection_name)
    qdrant_service.setup_collection(
        collection_name=collection_name,
        embedding_dimension=settings.EMBEDDING_DIMENSION,
        force_recreate=True,
        with_sparse_vectors=True,
    )

    logger.info("Step 4/4: Transferring points from staging into '%s' ...", collection_name)
    offset = None
    transferred = 0
    while True:
        records, next_offset = client.scroll(
            collection_name=staging_col,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not records:
            break

        transfer_points = [
            PointStruct(
                id=r.id,
                vector=r.vector,
                payload=r.payload or {},
            )
            for r in records
        ]

        client.upsert(
            collection_name=collection_name,
            points=transfer_points,
            wait=True,
        )

        transferred += len(records)
        offset = next_offset
        if offset is None:
            break

    # Clean up staging collection
    client.delete_collection(staging_col)

    # Invalidate cached schema in QdrantService
    if hasattr(qdrant_service, "_hybrid_supported_collections"):
        qdrant_service._hybrid_supported_collections.pop(collection_name, None)
    if hasattr(qdrant_service, "_dense_vector_collections"):
        qdrant_service._dense_vector_collections.pop(collection_name, None)

    final_info = client.get_collection(collection_name)
    logger.info(
        "🎉 SUCCESS: Collection '%s' migrated! Total points: %d (Hybrid enabled: %s)",
        collection_name, final_info.points_count, qdrant_service.collection_supports_hybrid(collection_name)
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Zero-Cost Hybrid Migration for Qdrant Collections.")
    parser.add_argument(
        "--collection",
        type=str,
        default="all",
        choices=["all", "quran", "hadith", "tafsir", "general_islamic_info"],
        help="Target collection to migrate (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for scrolling and upserting (default: 500)",
    )
    args = parser.parse_args()

    qdrant_service = QdrantService()

    logger.info("Loading fastembed BM25 sparse encoder (Qdrant/bm25)...")
    bm25_model = Bm25(model_name="Qdrant/bm25", language="english")
    logger.info("BM25 sparse encoder ready.")

    targets = (
        ["quran", "tafsir", "hadith"]
        if args.collection == "all"
        else [args.collection]
    )

    results = {}
    for col in targets:
        actual_name = settings.COLLECTION_NAMES.get(col, col)
        success = migrate_collection_to_hybrid(
            qdrant_service=qdrant_service,
            bm25_model=bm25_model,
            collection_name=actual_name,
            batch_size=args.batch_size,
        )
        results[actual_name] = success

    logger.info("=" * 60)
    logger.info("MIGRATION SUMMARY")
    logger.info("=" * 60)
    for col_name, succ in results.items():
        status = "MIGRATED / READY" if succ else "SKIPPED / ALREADY HYBRID"
        logger.info("%-25s : %s", col_name, status)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
