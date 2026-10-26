"""ReMem selected-frame task functions for lmms-eval."""

from __future__ import annotations

import os
import re
import importlib.util
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

import av
import imageio_ffmpeg
from PIL import Image

_LAST_VIDEO = None
_LAST_FRAMES = None
_LAST_FPS = None


def _extract_mcq_answer(response: str) -> str:
    """Load the official lmms-eval extractor without importing all task deps."""
    root = Path(os.environ.get("LMMS_EVAL_ROOT", "D:/lmms-eval-main"))
    path = root / "lmms_eval" / "tasks" / "_task_utils" / "mcq_extract.py"
    spec = importlib.util.spec_from_file_location("remem_lmms_mcq_extract", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official extractor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_mcq_answer(response, choices=["A", "B", "C", "D"])


def _probe(video_path: str) -> Tuple[float, int, int, int]:
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.base_rate
        fps = float(rate) if rate else 0.0
        count = int(stream.frames or 0)
        if count <= 0 and stream.duration and stream.time_base and fps > 0:
            count = max(1, int(round(float(stream.duration * stream.time_base) * fps)))
        width, height = int(stream.width), int(stream.height)
    if fps <= 0 or count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path}")
    return fps, count, width, height


def _decode(video_path: str, indices: List[int], width: int, height: int) -> List[Image.Image]:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    bytes_per_frame = width * height * 3
    with tempfile.TemporaryDirectory(prefix="remem-lmms-ffmpeg-") as temp_dir:
        script = Path(temp_dir) / "select.txt"
        script.write_text("select=" + "+".join(f"eq(n\\,{i})" for i in indices), encoding="utf-8")
        process = subprocess.Popen(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", video_path,
             "-filter_script:v", str(script), "-vsync", "0", "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        frames = []
        try:
            assert process.stdout is not None
            for _ in indices:
                data = bytearray()
                while len(data) < bytes_per_frame:
                    chunk = process.stdout.read(bytes_per_frame - len(data))
                    if not chunk:
                        break
                    data.extend(chunk)
                if len(data) != bytes_per_frame:
                    error = (process.stderr.read() if process.stderr else b"").decode("utf-8", "replace")[-500:]
                    raise RuntimeError(f"FFmpeg decoded {len(frames)}/{len(indices)} frames: {error}")
                frames.append(Image.frombytes("RGB", (width, height), bytes(data)))
            if process.wait(timeout=120) != 0:
                error = (process.stderr.read() if process.stderr else b"").decode("utf-8", "replace")[-500:]
                raise RuntimeError(f"FFmpeg failed: {error}")
            return frames
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()


def _decode_pyav(video_path: str, indices: List[int]) -> List[Image.Image]:
    """Fallback decoder for files where the isolated FFmpeg pipe returns no frames."""
    wanted = sorted({int(index) for index in indices})
    frames = []
    target = 0
    with av.open(video_path) as container:
        for frame_number, frame in enumerate(container.decode(video=0)):
            if target >= len(wanted):
                break
            if frame_number != wanted[target]:
                continue
            frames.append(frame.to_image().convert("RGB"))
            target += 1
    if target != len(wanted):
        raise RuntimeError(f"PyAV decoded {target}/{len(wanted)} frames: {video_path}")
    by_index = {index: image for index, image in zip(wanted, frames)}
    return [by_index[int(index)] for index in indices]


def _get_frames(doc: dict) -> Tuple[List[Image.Image], List[float]]:
    global _LAST_VIDEO, _LAST_FRAMES, _LAST_FPS
    video_path = str(doc["video_path"])
    fps, count, width, height = _probe(video_path)
    indices = sorted({min(max(int(float(index)), 0), count - 1) for index in doc.get("frame_idx", [])})
    if video_path == _LAST_VIDEO and _LAST_FRAMES is not None and _LAST_FPS == fps:
        # The three questions for a video usually share the selected frame set;
        # only reuse when the exact ordinal list also matches.
        if getattr(_get_frames, "_indices", None) == indices:
            return _LAST_FRAMES, [index / fps for index in indices]
    try:
        frames = _decode(video_path, indices, width, height)
    except Exception as error:
        print(
            f"FFmpeg decode failed for {doc.get('question_id', '')} "
            f"({video_path}); falling back to PyAV: {error}",
            flush=True,
        )
        frames = _decode_pyav(video_path, indices)
    _LAST_VIDEO, _LAST_FRAMES, _LAST_FPS = video_path, frames, fps
    _get_frames._indices = indices
    return frames, [index / fps for index in indices]


def remem_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "Question: ")
    post_prompt = kwargs.get("post_prompt", "Answer with the option letter only.")
    options = []
    for index, option in enumerate(doc["options"]):
        letter = chr(65 + index)
        text = str(option).strip()
        if not re.match(rf"^{re.escape(letter)}[.)]:?\s", text):
            text = f"{letter}. {text}"
        options.append(text)
    prompt = f"{pre_prompt}{doc['question']}\nOptions:\n{chr(10).join(options)}\n{post_prompt}"
    frames, timestamps = _get_frames(doc)
    # Keep the selected frames as one video item so the official Qwen3-VL
    # processor applies temporal video encoding and one consistent resize
    # policy.  Preserve the original ordinal-to-time mapping explicitly,
    # because the list form otherwise has synthetic indices 0..N-1.
    content = [{"type": "video", "url": frames}]
    content.append({
        "type": "text",
        "text": "Selected frame timestamps (seconds): " + ", ".join(f"{t:.3f}" for t in timestamps),
    })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def remem_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Plain-text fallback used by lmms_eval bookkeeping/cache setup.

    The actual Qwen3-VL request is built by ``remem_doc_to_messages`` so the
    selected images and timestamps remain in the multimodal payload.
    """
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "Question: ")
    post_prompt = kwargs.get("post_prompt", "Answer with the option letter only.")
    options = []
    for index, option in enumerate(doc["options"]):
        letter = chr(65 + index)
        text = str(option).strip()
        if not re.match(rf"^{re.escape(letter)}[.)]:?\s", text):
            text = f"{letter}. {text}"
        options.append(text)
    return f"{pre_prompt}{doc['question']}\nOptions:\n{chr(10).join(options)}\n{post_prompt}"


def remem_process_results(doc, results):
    pred = _extract_mcq_answer(results[0])
    expected = str(doc["answer"]).strip().upper()
    return {
        "videomme_perception_score": {
            "question_id": doc["question_id"],
            "duration": doc.get("duration", ""),
            "category": doc.get("domain", ""),
            "sub_category": doc.get("sub_category", ""),
            "task_category": doc.get("task_type", ""),
            "pred_answer": pred,
            "answer": expected,
            "score": float(pred.upper() == expected),
            "videoID": doc.get("videoID", ""),
        }
    }


def remem_aggregate_results(results):
    total = len(results)
    correct = sum(int(row.get("pred_answer", "").upper() == str(row.get("answer", "")).upper()) for row in results)
    return 100.0 * correct / total if total else 0.0
