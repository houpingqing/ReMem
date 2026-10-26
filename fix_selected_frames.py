"""Ensure every selected-frame record contains exactly 32 unique indices.

The released selector can return fewer than the requested budget for a small
number of records. This preserves all existing selected indices and fills the
remaining slots with uniformly spaced video indices.
"""

import argparse
import json
from pathlib import Path

from decord import VideoReader, cpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("video_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target", type=int, default=32)
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    changed = 0
    for i, record in enumerate(records, 1):
        selected = sorted({int(x) for x in record.get("frame_idx", [])})
        if len(selected) >= args.target:
            record["frame_idx"] = selected[: args.target]
        else:
            path = args.video_root / record["video_name"].replace("\\", "/")
            vr = VideoReader(str(path), ctx=cpu(0))
            uniform = [
                round(j * (len(vr) - 1) / max(args.target - 1, 1))
                for j in range(args.target)
            ]
            merged = sorted(set(selected).union(uniform))
            for index in merged:
                if len(selected) >= args.target:
                    break
                if index not in selected:
                    selected.append(index)
            record["frame_idx"] = sorted(selected)
            changed += 1
        if len(record["frame_idx"]) != args.target:
            raise RuntimeError(f"{record['question_id']} has {len(record['frame_idx'])} frames")
        print(f"Fixed {i}/{len(records)}: {record['question_id']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} records; padded {changed}", flush=True)


if __name__ == "__main__":
    main()
