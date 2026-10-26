"""Video-MME answering path compatible with lmms-eval's Qwen3-VL format.

The ReMem selector stores original video frame ordinals in ``frame_idx``.
This runner decodes exactly those ordinals with FFmpeg, then serializes the
request in the same format as lmms-eval's ``message_format=qwen3_vl`` path:
each image is preceded by a ``<N.N seconds>`` text item.  It is intentionally
resumable by question_id and records the actual frames/timestamps sent.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import av
import imageio_ffmpeg
from httpx2 import Limits
from openai import DefaultHttpxClient, OpenAI
from PIL import Image


def _probe_video(video_path: Path) -> Tuple[float, int, int, int]:
    """Return fps, frame count, width and height using the same FFmpeg stack."""
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate
        fps = float(rate) if rate else 0.0
        frame_count = int(stream.frames or 0)
        if frame_count <= 0 and stream.duration and stream.time_base and fps > 0:
            frame_count = max(1, int(round(float(stream.duration * stream.time_base) * fps)))
        width, height = int(stream.width), int(stream.height)
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Cannot read video metadata: {video_path}")
    return fps, frame_count, width, height


def _ffmpeg_decode(video_path: Path, indices: Sequence[int], width: int, height: int) -> List[Image.Image]:
    """Decode selected frame ordinals in a separate FFmpeg process."""
    if not indices:
        return []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    frame_bytes = width * height * 3
    with tempfile.TemporaryDirectory(prefix="remem-answer-ffmpeg-") as temp_dir:
        select_script = Path(temp_dir) / "select.txt"
        # This is the same ordinal selector used by frame_select.py.  Keeping
        # it in a subprocess avoids Windows Decord/native H264 crashes.
        select_script.write_text(
            "select=" + "+".join(f"eq(n\\,{int(i)})" for i in indices),
            encoding="utf-8",
        )
        proc = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video_path),
                "-filter_script:v",
                str(select_script),
                "-vsync",
                "0",
                "-an",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        frames: List[Image.Image] = []
        try:
            assert proc.stdout is not None
            for _ in indices:
                raw = bytearray()
                while len(raw) < frame_bytes:
                    piece = proc.stdout.read(frame_bytes - len(raw))
                    if not piece:
                        break
                    raw.extend(piece)
                if len(raw) != frame_bytes:
                    stderr = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", "replace")
                    raise RuntimeError(
                        f"FFmpeg decoded {len(frames)}/{len(indices)} frames for {video_path}: {stderr[-500:]}"
                    )
                frames.append(Image.frombytes("RGB", (width, height), bytes(raw)))
            code = proc.wait(timeout=120)
            if code != 0:
                stderr = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", "replace")
                raise RuntimeError(f"FFmpeg exited {code} for {video_path}: {stderr[-500:]}")
            return frames
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()


def _encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compress_level=1)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _qwen3_vl_messages(images: Sequence[Image.Image], timestamps: Sequence[float], prompt: str) -> List[dict]:
    """Serialize the lmms-eval Qwen3-VL message format without extra deps."""
    content = []
    for image, timestamp in zip(images, timestamps):
        content.append({"type": "text", "text": f"<{timestamp:.1f} seconds>"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{_encode_png(image)}"},
            }
        )
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _format_prompt(record: dict) -> str:
    # Exact qwen3_vl branch of lmms_eval/tasks/videomme/utils.py.
    options = []
    for index, choice in enumerate(record["candidates"]):
        letter = chr(65 + index)
        text = str(choice).strip()
        if not re.match(rf"^{re.escape(letter)}[.)]:?\s", text):
            text = f"{letter}. {text}"
        options.append(text)
    return (
        f"Question: {record['question']}\n"
        f"Options:\n{chr(10).join(options)}\n"
        "Answer with the option letter only."
    )


def _load_results(path: Path, resume: bool) -> List[dict]:
    if resume and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    if args.limit is not None:
        labels = labels[: args.limit]
    output = Path(args.output)
    results = _load_results(output, args.resume)
    completed = {str(row["question_id"]) for row in results}
    video_root = Path(args.video_root)
    video_cache: Dict[str, Tuple[float, List[Image.Image]]] = {}

    client = OpenAI(
        base_url=os.getenv("REMEM_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        timeout=600,
        max_retries=2,
        http_client=DefaultHttpxClient(
            trust_env=False,
            limits=Limits(max_keepalive_connections=0),
        ),
    )
    model = os.getenv("REMEM_LLM_MODEL", "qwen3-vl-8b-instruct")

    for number, record in enumerate(labels, 1):
        qid = str(record["question_id"])
        if qid in completed:
            continue
        video_name = str(record["video_name"]).replace("\\", "/")
        video_path = video_root / video_name
        requested = sorted({max(0, int(float(index))) for index in record.get("frame_idx", [])})
        fps, frame_count, width, height = _probe_video(video_path)
        requested = [min(index, frame_count - 1) for index in requested]
        requested = list(dict.fromkeys(requested))
        if len(requested) > args.max_frames:
            requested = [
                requested[round(i * (len(requested) - 1) / max(args.max_frames - 1, 1))]
                for i in range(args.max_frames)
            ]
            requested = list(dict.fromkeys(requested))
        key = str(video_path.resolve())
        if key not in video_cache:
            # Cache only the complete video decode once; the same VideoMME
            # video has three questions and identical selected frame ordinals.
            video_cache[key] = (fps, [])
        print(
            f"Preparing {number}/{len(labels)}: {qid} frames={len(requested)} fps={fps:.6g}",
            flush=True,
        )
        images = _ffmpeg_decode(video_path, requested, width, height)
        timestamps = [index / fps for index in requested]
        if len(images) != len(timestamps):
            raise RuntimeError(f"Frame count mismatch for {qid}: {len(images)} vs {len(timestamps)}")
        prompt = _format_prompt(record)
        messages = _qwen3_vl_messages(images, timestamps, prompt)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=16,
            temperature=0,
        )
        prediction = response.choices[0].message.content or ""
        results.append(
            {
                "question_id": record["question_id"],
                "prediction": prediction,
                "frame_idx": requested,
                "frame_timestamps": timestamps,
                "decoder": "ffmpeg",
                "fps": fps,
            }
        )
        completed.add(qid)
        temporary = output.with_suffix(output.suffix + ".tmp")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)
        print(f"Answered {number}/{len(labels)}: {qid} ({len(requested)} frames)", flush=True)

    print(f"Completed {len(results)}/{len(labels)} questions: {output}", flush=True)


if __name__ == "__main__":
    main()
