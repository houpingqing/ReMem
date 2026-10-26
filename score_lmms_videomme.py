"""Score ReMem predictions with lmms-eval's official MCQ extractor."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def load_lmms_extractor() -> object:
    root = Path(__file__).resolve().parent
    path = root.parent / "lmms-eval-main" / "lmms_eval" / "tasks" / "_task_utils" / "mcq_extract.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("lmms_mcq_extract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_mcq_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parquet", default="VideoMME/test-00000-of-00001.parquet")
    args = parser.parse_args()
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    by_id = {str(row["question_id"]): row for row in predictions}
    extract = load_lmms_extractor()
    metadata = {}
    parquet = Path(args.parquet)
    if parquet.is_file():
        import pyarrow.parquet as pq

        metadata = {str(row["question_id"]): row for row in pq.read_table(parquet).to_pylist()}
    buckets = {"overall": [], "short": [], "medium": [], "long": []}
    rows = []
    for label in labels:
        qid = str(label["question_id"])
        prediction = by_id.get(qid, {}).get("prediction", "")
        actual = extract(prediction, choices=["A", "B", "C", "D"])
        expected = str(label["answer"]).strip().upper()
        correct = int(actual.upper() == expected)
        duration = metadata.get(qid, {}).get("duration", label.get("duration"))
        duration_key = duration if duration in {"short", "medium", "long"} else None
        row = {"question_id": label["question_id"], "prediction": prediction, "parsed": actual, "answer": expected, "correct": correct, "duration": duration}
        rows.append(row)
        buckets["overall"].append(correct)
        if duration_key:
            buckets[duration_key].append(correct)

    metrics = {}
    for key, values in buckets.items():
        metrics[key] = {"correct": sum(values), "total": len(values), "accuracy": (sum(values) / len(values) if values else 0.0)}
    output = {"metrics": metrics, "coverage": {"labels": len(labels), "predictions": len(predictions), "unique_predictions": len(by_id)}, "rows": rows}
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
