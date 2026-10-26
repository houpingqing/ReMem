"""Answer selected-frame video questions with an OpenAI-compatible vLLM server."""

import argparse
import base64
import io
import json
import os
from pathlib import Path

from decord import VideoReader, cpu
from httpx2 import Limits
from openai import DefaultHttpxClient, OpenAI
from PIL import Image


def image_url(frame, max_side=0):
    image = Image.fromarray(frame.asnumpy())
    if max_side:
        image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="./VideoMME/val_qa_selected_full_32f.json")
    parser.add_argument("--video-root", default="./VideoMME/videos")
    parser.add_argument("--output", default="./VideoMME/vllm_predictions.json")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--image-max-side", type=int, default=0, help="0 keeps the source frame resolution")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted output file")
    parser.add_argument("--protocol", choices=("legacy", "videomme_qwen3_vl"), default="legacy")
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.image_max_side < 0 or 0 < args.image_max_side < 32:
        parser.error("--image-max-side must be 0 or at least 32")
    client = OpenAI(
        base_url=os.getenv("REMEM_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        timeout=180,
        max_retries=0,
        http_client=DefaultHttpxClient(trust_env=False, limits=Limits(max_keepalive_connections=0)),
    )
    model = os.getenv("REMEM_LLM_MODEL", "qwen3-vl-4b-instruct")
    records = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    if args.limit is not None:
        records = records[:args.limit]
    output = Path(args.output)
    results = json.loads(output.read_text(encoding="utf-8")) if args.resume and output.exists() else []
    completed = {str(row["question_id"]) for row in results}
    for i, record in enumerate(records, 1):
        if str(record.get("question_id")) in completed:
            continue
        print(f"Preparing {i}/{len(records)}: {record.get('question_id')}", flush=True)
        vr = VideoReader(str(Path(args.video_root) / record["video_name"].replace("\\", "/")), ctx=cpu(0))
        indices = sorted({min(max(int(n), 0), len(vr) - 1) for n in record["frame_idx"]})
        if len(indices) > args.max_frames:
            indices = [indices[round(j * (len(indices) - 1) / max(args.max_frames - 1, 1))] for j in range(args.max_frames)]
        choices = "\n".join(choice if str(choice).startswith(f"{chr(65+j)}.") else f"{chr(65+j)}. {choice}" for j, choice in enumerate(record["candidates"]))
        if args.protocol == "videomme_qwen3_vl":
            prompt = f"Question: {record['question']}\nOptions:\n{choices}\nAnswer with the option letter only."
            content = [{"type": "image_url", "image_url": {"url": image_url(vr[index], args.image_max_side)}} for index in indices]
            content.append({"type": "text", "text": prompt})
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ]
            print(f"Sending {i}/{len(records)} with {len(indices)} frames", flush=True)
            response = client.chat.completions.create(
                model=model, messages=messages, max_tokens=16, temperature=0, top_p=1.0
            )
        else:
            content = [{"type": "text", "text": f"Answer the video question using the frames in chronological order. Reply with only the answer letter.\nQuestion: {record['question']}\n{choices}"}]
            content.extend({"type": "image_url", "image_url": {"url": image_url(vr[index], args.image_max_side)}} for index in indices)
            response = client.chat.completions.create(model=model, messages=[{"role": "user", "content": content}], max_tokens=32)
        results.append({"question_id": record.get("question_id"), "prediction": response.choices[0].message.content, "frame_idx": indices})
        print(f"Answered {i}/{len(records)} with {len(indices)} frames", flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)


if __name__ == "__main__":
    main()
