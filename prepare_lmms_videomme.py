"""Convert ReMem selected annotations to a local lmms-eval JSONL task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", required=True)
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selected = json.loads(Path(args.selected).read_text(encoding="utf-8"))
    import pyarrow.parquet as pq

    meta = {str(row["question_id"]): row for row in pq.read_table(args.parquet).to_pylist()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in selected:
            qid = str(record["question_id"])
            source = meta.get(qid, {})
            row = dict(record)
            row["video_path"] = str((Path("D:/ReMem/VideoMME/videos") / record["video_name"].replace("\\", "/")).resolve())
            row["options"] = record["candidates"]
            row["videoID"] = source.get("videoID", Path(record["video_name"]).stem)
            row["duration"] = source.get("duration", "short")
            row["domain"] = source.get("domain", "")
            row["sub_category"] = source.get("sub_category", "")
            row["task_type"] = source.get("task_type", "")
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected)} lmms-eval documents: {output}")


if __name__ == "__main__":
    main()
