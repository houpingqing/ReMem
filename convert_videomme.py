"""Convert the official Video-MME annotation parquet to ReMem JSON."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from decord import VideoReader, cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("video_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    videos = {path.stem: path for path in args.video_root.rglob("*.mp4")}
    records = []
    missing = set()
    durations = {}
    for row in pq.read_table(args.parquet).to_pylist():
        video = videos.get(str(row["videoID"])) or videos.get(str(row["video_id"]))
        if video is None:
            missing.add(str(row["videoID"]))
            continue
        if video not in durations:
            vr = VideoReader(str(video), ctx=cpu(0))
            durations[video] = len(vr) / vr.get_avg_fps()
        records.append({
            "question_id": row["question_id"],
            "video_name": str(video.relative_to(args.video_root)),
            "question": row["question"],
            "candidates": row["options"],
            "answer": row["answer"],
            "duration": durations[video],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} questions to {args.output}")
    if missing:
        print(f"Skipped {len(missing)} videos missing from {args.video_root}")


if __name__ == "__main__":
    main()
