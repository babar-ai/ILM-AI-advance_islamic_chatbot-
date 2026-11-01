"""Run RAGAS evaluation for exactly 40 questions loaded from a JSON file.

Accepted input formats are a JSON array of question strings or an object with
a ``questions`` array. Results are appended to a CSV file.

Usage:
    python evaluation/run_json_evaluation.py questions.json
    python evaluation/run_json_evaluation.py questions.json --output results.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "json_batch_scores.csv"
FIELDNAMES = [
    "timestamp",
    "question",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "contexts_count",
    "sources_used",
    "latency_ms",
    "error",
    "answer",
]


def load_questions(input_path: Path) -> List[str]:
    """Load and validate exactly 40 non-empty question strings."""
    with input_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        raise ValueError("JSON must be an array of questions or an object with a 'questions' array.")
    if len(questions) != 40:
        raise ValueError(f"Expected exactly 40 questions; found {len(questions)}.")
    if any(not isinstance(question, str) or not question.strip() for question in questions):
        raise ValueError("Every question must be a non-empty string.")

    return [question.strip() for question in questions]


def append_result(output_path: Path, result: dict) -> None:
    """Append one evaluation result, writing a header for a new CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    row = dict(result)
    row["sources_used"] = ", ".join(row.get("sources_used", []))

    with output_path.open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in FIELDNAMES})


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 40 JSON questions with RAGAS.")
    parser.add_argument("questions_json", type=Path, help="JSON array or {'questions': [...]} with 40 strings.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Append results to this CSV.")
    args = parser.parse_args()

    try:
        questions = load_questions(args.questions_json)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    from evaluation.run_evaluation import evaluate_question, print_summary
    from evaluation.pipeline_tracer import TracingLangGraphService

    service = TracingLangGraphService()
    results = []
    try:
        for index, question in enumerate(questions, start=1):
            print(f"\n[{index}/40] Evaluating question")
            result = evaluate_question(service, question)
            append_result(args.output, result)
            results.append(result)
    finally:
        service.classification_executor.shutdown(wait=True)
        service.retrival_executor.shutdown(wait=True)

    print(f"\nSaved 40 evaluation rows to: {args.output.resolve()}")
    print_summary(results)


if __name__ == "__main__":
    main()
