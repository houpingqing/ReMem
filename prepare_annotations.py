"""Add ReMem question fields to a Video-MME-style annotation JSON."""

import argparse
import json
from pathlib import Path

from entity_extraction import classify_question as extract_entities
from granularity_analysis import classify_question as classify_granularity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Annotation JSON must be a list of records")
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        by_id = {str(record["question_id"]): record for record in previous}
        if len(by_id) != len(previous):
            raise ValueError("Duplicate question_id in existing output")
        for record in records:
            saved = by_id.get(str(record["question_id"]))
            if saved is not None:
                for field in ("granularity", "entity_keywords"):
                    if field in saved:
                        record[field] = saved[field]
    for i, record in enumerate(records, 1):
        if "granularity" not in record:
            record["granularity"] = float(classify_granularity(record["question"]))
        if not record.get("entity_keywords"):
            record["entity_keywords"] = extract_entities(record["question"], record["candidates"])
        print(f"Prepared {i}/{len(records)}: {record.get('question_id', '')}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
