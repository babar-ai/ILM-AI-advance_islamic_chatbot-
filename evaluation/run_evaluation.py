"""
run_evaluation.py
-----------------
Interactive evaluation runner for the Islamic Chatbot.

Usage:
    python evaluation/run_evaluation.py

How it works:
    1. Type any question at the prompt
    2. It runs through the full LangGraph pipeline
       (classify_and_search → parallel_retrieve → generate_response)
    3. RAGAS computes: Faithfulness, Answer Relevancy, Context Relevancy
    4. If LangSmith is configured, scores are pushed as feedback to the run trace
    5. Results are printed to console and saved to evaluation/results/scores.csv
"""

import sys
import csv
from pathlib import Path
from datetime import datetime

# Prevent Windows console charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pipeline_tracer import TracingLangGraphService
from evaluation.ragas_evaluator import run_ragas
from evaluation.langsmith_feedback import push_scores_to_langsmith, get_latest_run_id
from utils.config import settings
from utils.custom_logger import setup_logger

logger = setup_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
# Always save relative to project root, regardless of where the script is run from
OUTPUT_CSV = ROOT / "evaluation" / "results" / "scores.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_question(
    service: TracingLangGraphService,
    question: str,
) -> dict:
    """
    Runs one question through the full pipeline and evaluates it with RAGAS.
    Returns a dict with question, answer, contexts, scores, and metadata.
    """
    print(f"\n🔄  Running pipeline for: \"{question[:80]}{'...' if len(question) > 80 else ''}\"")

    # Step 1: Run pipeline and capture full trace
    trace = service.query_with_trace(question)

    if trace.error:
        print(f"❌  Pipeline error: {trace.error}")
        return {
            "question": question,
            "answer": "",
            "contexts_count": 0,
            "sources_used": [],
            "latency_ms": trace.latency_ms,
            "faithfulness": None,
            "answer_relevancy": None,
            "context_relevancy": None,
            "error": trace.error,
            "timestamp": datetime.now().isoformat(),
        }

    print(f"✅  Pipeline complete | sources={trace.sources_used} | contexts={len(trace.contexts)} | {trace.latency_ms:.0f}ms")

    # Step 2: Run RAGAS evaluation
    print("📊  Running RAGAS evaluation...")
    scores = run_ragas(
        question=trace.question,
        answer=trace.answer,
        contexts=trace.contexts,
        verbose=True,
    )

    # Step 3: Push scores to LangSmith (if configured)
    if settings.langsmith_enabled and scores:
        run_id = get_latest_run_id()
        if run_id:
            pushed = push_scores_to_langsmith(run_id=run_id, scores=scores)
            if pushed:
                print(f"📤  RAGAS scores pushed to LangSmith run {run_id[:8]}...")

    res = {
        "question": question,
        "answer": trace.answer[:200] + "..." if len(trace.answer) > 200 else trace.answer,
        "contexts_count": len(trace.contexts),
        "sources_used": trace.sources_used,
        "latency_ms": trace.latency_ms,
        "faithfulness": scores.get("faithfulness") if scores else None,
        "answer_relevancy": scores.get("answer_relevancy") if scores else None,
        "context_precision": scores.get("context_precision") if scores else None,
        "error": None,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(
        "Evaluation Result Logged | question='%s' | sources=%s | contexts=%d | latency=%.0fms | faithfulness=%s | answer_relevancy=%s | context_precision=%s",
        question[:80],
        trace.sources_used,
        len(trace.contexts),
        trace.latency_ms,
        res["faithfulness"],
        res["answer_relevancy"],
        res["context_precision"],
    )

    return res


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_to_csv(result: dict) -> None:
    """Append a single evaluation result to the CSV log file."""
    path = Path(OUTPUT_CSV)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp", "question", "faithfulness", "answer_relevancy",
        "context_precision", "contexts_count", "sources_used",
        "latency_ms", "error", "answer",
    ]

    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        result["sources_used"] = ", ".join(result.get("sources_used", []))
        writer.writerow({k: result.get(k, "") for k in fieldnames})

    print(f"💾  Saved to: {path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    valid = [r for r in results if r["faithfulness"] is not None]
    if not valid:
        print("\n⚠️  No valid results to summarize.")
        return

    avg_faith = sum(r["faithfulness"] for r in valid) / len(valid)
    avg_rel   = sum(r["answer_relevancy"] for r in valid) / len(valid)
    c_prec_vals = [r.get("context_precision") for r in valid if r.get("context_precision") is not None]
    avg_prec  = sum(c_prec_vals) / len(c_prec_vals) if c_prec_vals else 0.0

    logger.info(
        "Evaluation Session Summary | total_questions=%d | avg_faithfulness=%.4f | avg_answer_relevancy=%.4f | avg_context_precision=%.4f",
        len(valid),
        avg_faith,
        avg_rel,
        avg_prec,
    )

    print("\n" + "=" * 60)
    print("📈  OVERALL SESSION SUMMARY")
    print(f"    Questions evaluated : {len(valid)}")
    print("-" * 60)
    print(f"    Faithfulness        : {avg_faith:.4f}")
    print(f"    Answer Relevancy    : {avg_rel:.4f}")
    print(f"    Context Precision   : {avg_prec:.4f}")
    print("=" * 60)


BENCHMARK_QUESTIONS = [
    "what are the five pillars of islam?",
    "what is zakat? and why it is compuslory for muslims to pay?",
    "What does Surah Al-Baqarah verse 153 say about patience and prayer?",
    "What does Sahih al-Bukhari say about intentions and deeds?",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main — interactive loop & CLI runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Islamic Chatbot — RAGAS Evaluation Runner")
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Evaluate a single question directly without interactive prompts.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a standard multi-source benchmark covering Quran, Hadith, Tafsir, and General Islamic Info.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  🕌  Islamic Chatbot — RAGAS Evaluation")
    print(f"  LangSmith: {'✅ Enabled' if settings.langsmith_enabled else '❌ Disabled (add LANGCHAIN_API_KEY to .env)'}")
    print("=" * 60)

    print("\n⏳  Initializing pipeline service...")
    service = TracingLangGraphService()
    print("✅  Pipeline ready.\n")

    results = []

    # Mode 1: Single question passed via CLI
    if args.question:
        result = evaluate_question(service, args.question.strip())
        results.append(result)
        save_to_csv(result)
        print_summary(results)
        return

    # Mode 2: Multi-source benchmark
    if args.benchmark:
        print(f"🚀  Running benchmark with {len(BENCHMARK_QUESTIONS)} curated questions...\n")
        for idx, q in enumerate(BENCHMARK_QUESTIONS, 1):
            print(f"\n[{idx}/{len(BENCHMARK_QUESTIONS)}] Benchmark Question: {q}")
            result = evaluate_question(service, q)
            results.append(result)
            save_to_csv(result)
        print_summary(results)
        return

    # Mode 3: Interactive prompt
    print("💬  Type your question and press Enter to evaluate.")
    print("    Type 'quit' or press Ctrl+C to exit.\n")

    while True:
        try:
            question = input("❓  Your question: ").strip()

            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue

            result = evaluate_question(service, question)
            results.append(result)
            save_to_csv(result)

            again = input("\n   Evaluate another question? (y/n): ").strip().lower()
            if again != "y":
                break

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
