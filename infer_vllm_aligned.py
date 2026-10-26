"""Run a controlled selected-frame Video-MME evaluation.

This keeps the existing ReMem frame_idx file and generation settings unchanged
while changing only the image transport (lossless PNG) and message packaging.
Results are written atomically and can be resumed by question_id.
"""

import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

from decord import VideoReader, cpu
from httpx2 import Limits
from openai import DefaultHttpxClient, OpenAI
from PIL import Image


def image_url(frame):
    image = Image.fromarray(frame.asnumpy())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    client = OpenAI(
        base_url=os.getenv("REMEM_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        timeout=300,
        max_retries=2,
        http_client=DefaultHttpxClient(
            trust_env=False,
            limits=Limits(max_keepalive_connections=0),
        ),
    )
    model = os.getenv("REMEM_LLM_MODEL", "qwen3-vl-8b-instruct")
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    output = Path(args.output)
    results = (
        json.loads(output.read_text(encoding="utf-8"))
        if args.resume and output.exists()
        else []
    )
    completed = {str(row["question_id"]) for row in results}
    video_root = Path(args.video_root)

    for i, record in enumerate(labels, 1):
        question_id = str(record["question_id"])
        if question_id in completed:
            continue
        video_path = video_root / record["video_name"].replace("\\", "/")
        print(f"Preparing {i}/{len(labels)}: {question_id}", flush=True)
        vr = VideoReader(str(video_path), ctx=cpu(0))
        indices = sorted(
            {min(max(int(n), 0), len(vr) - 1) for n in record["frame_idx"]}
        )
        if len(indices) > args.max_frames:
            indices = [
                indices[round(j * (len(indices) - 1) / max(args.max_frames - 1, 1))]
                for j in range(args.max_frames)
            ]
        choices = "\n".join(
            choice if str(choice).startswith(f"{chr(65 + j)}.")
            else f"{chr(65 + j)}. {choice}"
            for j, choice in enumerate(record["candidates"])
        )
        prompt = (
            f"Question: {record['question']}\n"
            f"Options:\n{choices}\n"
            "Answer with the option letter only."
        )
        content = [
            {"type": "image_url", "image_url": {"url": image_url(vr[index])}}
            for index in indices
        ]
        content.append({"type": "text", "text": prompt})
        print(f"Sending {i}/{len(labels)} with {len(indices)} frames", flush=True)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=16,
            temperature=0,
            top_p=1.0,
        )
        results.append(
            {
                "question_id": record["question_id"],
                "prediction": response.choices[0].message.content,
                "frame_idx": indices,
            }
        )
        completed.add(question_id)
        print(f"Answered {i}/{len(labels)} with {len(indices)} frames", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)


if __name__ == "__main__":
    main()
