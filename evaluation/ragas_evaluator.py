"""
ragas_evaluator.py
------------------
Runs RAGAS reference-free metrics on a PipelineTrace captured from the
live Islamic chatbot pipeline.

Metrics used (all reference-free — no ground truth needed):
  - faithfulness       : Are all claims supported by retrieved context?
  - answer_relevancy   : Does the answer actually address the question?
  - context_precision  : How many retrieved chunks are truly useful?
"""

import sys
import math
import types
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.custom_logger import setup_logger
from utils.config import settings

logger = setup_logger(__name__)

# ── Compatibility Shim ────────────────────────────────────────────────────────
# ragas <= 0.2.x imports langchain_community.chat_models.vertexai in ragas.llms.base,
# which was removed in langchain-community >= 0.3. Stub it once at import time.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    stub = types.ModuleType("langchain_community.chat_models.vertexai")
    stub.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = stub

# ── Top-Level Imports (Loaded once) ──────────────────────────────────────────
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, LLMContextPrecisionWithoutReference
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    RAGAS_AVAILABLE = True

except ImportError as e:
    logger.warning("RAGAS or its dependencies are not installed: %s. Run: pip install ragas datasets", e)
    RAGAS_AVAILABLE = False
    Dataset = None
    evaluate = None
    faithfulness = None
    answer_relevancy = None
    LLMContextPrecisionWithoutReference = None
    ChatOpenAI = None
    OpenAIEmbeddings = None
    ChatGroq = None
    HuggingFaceEmbeddings = None

# Reusable judge instances (cached to avoid re-instantiating HTTP clients per call)
_JUDGE_LLM: Optional[Any] = None
_JUDGE_EMBEDDINGS: Optional[Any] = None
_METRIC_CONTEXT_PREC: Optional["LLMContextPrecisionWithoutReference"] = None


def _get_judge_components() -> Tuple[Optional[Any], Optional[Any], Optional["LLMContextPrecisionWithoutReference"]]:
    """Initializes and caches the judge LLM, embeddings, and metrics once."""
    global _JUDGE_LLM, _JUDGE_EMBEDDINGS, _METRIC_CONTEXT_PREC
    if not RAGAS_AVAILABLE:
        return None, None, None

    if _JUDGE_LLM is None:
        if settings.USE_PRIMARY_LLM and settings.OPENAI_API_KEY:
            judge_model = getattr(settings, "EVAL_LLM_MODEL", settings.LLM_MODEL)
            _JUDGE_LLM = ChatOpenAI(
                model=judge_model,
                temperature=0,
                api_key=settings.OPENAI_API_KEY,
            )
        elif settings.GROQ_API_KEY and ChatGroq:
            logger.info("Using Groq (%s) as RAGAS Judge LLM.", settings.GROQ_LLM_MODEL)
            _JUDGE_LLM = ChatGroq(
                model=settings.GROQ_LLM_MODEL,
                temperature=0,
                api_key=settings.GROQ_API_KEY,
            )

    if _JUDGE_EMBEDDINGS is None:
        if getattr(settings, "USE_PRIMARY_EMBEDDING", True) and settings.OPENAI_API_KEY:
            _JUDGE_EMBEDDINGS = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                api_key=settings.OPENAI_API_KEY,
            )
        elif HuggingFaceEmbeddings:
            logger.info("Using HuggingFace (%s) as RAGAS Judge Embeddings.", settings.FALLBACK_EMBEDDING_MODEL)
            hf_kwargs = {"trust_remote_code": True}
            if settings.HF_TOKEN:
                hf_kwargs["token"] = settings.HF_TOKEN
            _JUDGE_EMBEDDINGS = HuggingFaceEmbeddings(
                model_name=settings.FALLBACK_EMBEDDING_MODEL,
                model_kwargs=hf_kwargs,
                encode_kwargs={"normalize_embeddings": True}
            )

    if _METRIC_CONTEXT_PREC is None:
        _METRIC_CONTEXT_PREC = LLMContextPrecisionWithoutReference()

    return _JUDGE_LLM, _JUDGE_EMBEDDINGS, _METRIC_CONTEXT_PREC


def _clean_score(val) -> float:
    """Sanitizes raw metric values from RAGAS, safely handling NaN and None."""
    if val is None:
        return 0.0
    try:
        num = float(val)
        return 0.0 if math.isnan(num) else round(num, 4)
    except (TypeError, ValueError):
        return 0.0


def run_ragas(
    question: str,
    answer: str,
    contexts: list[str],
    verbose: bool = True,
) -> Optional[Dict[str, float]]:
    """
    Runs RAGAS reference-free evaluation on a single question/answer/contexts triple.

    Args:
        question : The user's original question.
        answer   : The chatbot's generated answer.
        contexts : List of retrieved context strings.
        verbose  : Print clean summary to console if True.

    Returns:
        dict of metric_name -> float score (0.0 - 1.0), or None if RAGAS unavailable.
    """
    if not RAGAS_AVAILABLE:
        logger.error("RAGAS is not available. Please install dependencies: pip install ragas datasets")
        return None

    if not answer or not answer.strip():
        logger.warning("Empty answer provided — skipping RAGAS evaluation.")
        return None

    # Edge Case: If no contexts were retrieved, avoid paying for LLM calls and returning NaNs
    if not contexts:
        logger.warning("No contexts retrieved — returning zeroed reference-free scores without invoking LLM.")
        scores = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
        }
        if verbose:
            _print_scores(question, scores)
        return scores

    judge_llm, judge_embeddings, metric_context_prec = _get_judge_components()
    
    if not judge_llm or not judge_embeddings:
        logger.error("Failed to initialize judge models.")
        return None

    try:
        # Standard RAGAS 0.2 dataset format
        dataset = Dataset.from_dict({
            "user_input": [question],
            "response": [answer],
            "retrieved_contexts": [contexts],
        })
 
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, metric_context_prec],
            llm=judge_llm,
            embeddings=judge_embeddings,
            raise_exceptions=False,
        )

        df = result.to_pandas()
        row = df.to_dict(orient="records")[0] if not df.empty else {}

        scores = {
            "faithfulness": _clean_score(row.get("faithfulness")),
            "answer_relevancy": _clean_score(row.get("answer_relevancy")),
            "context_precision": _clean_score(
                row.get("llm_context_precision_without_reference", row.get("context_precision"))
            ),
        }

        logger.info(
            "RAGAS Evaluation Scores | question='%s' | faithfulness=%.4f | answer_relevancy=%.4f | context_precision=%.4f",
            question[:80],
            scores["faithfulness"],
            scores["answer_relevancy"],
            scores["context_precision"],
        )

        if verbose:
            _print_scores(question, scores)

        return scores

    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", e)
        return None


def _print_scores(question: str, scores: Dict[str, float]) -> None:
    """Prints a clean, concise, cross-platform summary of evaluation scores."""
    print("\n" + "=" * 50)
    print(f"RAGAS Scores for: {question[:60]}{'...' if len(question) > 60 else ''}")
    print("-" * 50)
    for metric, score in scores.items():
        label = metric.replace("_", " ").title()
        print(f"  {label:<22} : {score:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    print("Running standalone RAGAS evaluation test...\n")
    sample_question = "Why is interest (riba) forbidden in Islam?"
    sample_answer = (
        "In Islam, interest (riba) is strictly prohibited as stated in Surah Al-Baqarah (2:275). "
        "It is considered unjust because it involves gaining money without productive labor or sharing risk, "
        "leading to exploitation and economic inequality."
    )
    sample_contexts = [
        "Surah Al-Baqarah (2:275): 'Allah has permitted trade and forbidden usury (riba).' Usury exploits debtors.",
        "Islamic economic principles require transactions to be based on risk-sharing and mutual consent rather than guaranteed fixed returns on loans.",
    ]

    scores = run_ragas(
        question=sample_question,
        answer=sample_answer,
        contexts=sample_contexts,
        verbose=True,
    )
    print("Evaluation completed successfully.")

