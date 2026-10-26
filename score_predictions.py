"""Score local vLLM predictions against the downloaded Video-MME annotations."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def answer_letter(value):
    text = str(value or "").strip().upper()
    match = re.search(r"\b([A-D])\b", text)
    return match.group(1) if match else None


def summarize(rows):
    total = len(rows)
    correct = sum(row["correct"] for row in rows)
    return {"correct": correct, "total": total, "accuracy": correct / total if total else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("VideoMME/val_qa_full.json"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parquet", type=Path, default=Path("VideoMME/test-00000-of-00001.parquet"))
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    references = {str(row["question_id"]): row for row in labels}
    if len(references) != len(labels):
        parser.error("Duplicate question_id in labels")
    metadata = {}
    if args.parquet.is_file():
        import pyarrow.parquet as pq
        metadata = {str(row["question_id"]): row for row in pq.read_table(args.parquet).to_pylist()}

    seen = set()
    rows = []
    for prediction in predictions:
        question_id = str(prediction["question_id"])
        if question_id in seen:
            parser.error(f"Duplicate prediction for {question_id}")
        if question_id not in references:
            parser.error(f"Prediction has unknown question_id: {question_id}")
        seen.add(question_id)
        expected = answer_letter(references[question_id]["answer"])
        actual = answer_letter(prediction.get("prediction"))
        if expected is None:
            parser.error(f"Invalid reference answer for {question_id}")
        source = metadata.get(question_id, {})
        rows.append({
            "question_id": question_id,
            "expected": expected,
            "prediction": actual,
            "raw_prediction": prediction.get("prediction"),
            "correct": actual == expected,
            "duration": source.get("duration"),
            "domain": source.get("domain"),
            "task_type": source.get("task_type"),
        })

    result = {
        "overall": summarize(rows),
        "coverage": {"predicted": len(rows), "available": len(labels)},
        "invalid_predictions": sum(row["prediction"] is None for row in rows),
        "groups": {},
        "details": rows,
    }
    for field in ("duration", "domain", "task_type"):
        groups = defaultdict(list)
        for row in rows:
            if row[field] is not None:
                groups[str(row[field])].append(row)
        result["groups"][field] = {name: summarize(items) for name, items in sorted(groups.items())}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    overall = result["overall"]
    print(f"Accuracy: {overall['correct']}/{overall['total']} = {overall['accuracy']:.2%}" if overall["total"] else "No predictions to score")
    print(f"Coverage: {len(rows)}/{len(labels)} available questions")
    for duration, values in result["groups"]["duration"].items():
        print(f"  {duration}: {values['correct']}/{values['total']} = {values['accuracy']:.2%}")
    print(f"Detailed metrics: {args.output}")


if __name__ == "__main__":
    main()
