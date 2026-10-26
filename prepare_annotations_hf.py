"""Run the ReMem question parser with the local Qwen3-8B HF model.

This is the serial, non-vLLM replacement for the old OpenAI-compatible
annotation client.  Each record is written after it is processed so a crash
does not discard completed annotations.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ENTITY_TEMPLATE = """You are an expert in visual perception and long-term knowledge reasoning.
Task: Generate 3-5 high-quality \"Visual Entities\" (keywords) to help a video model locate the specific frames needed to distinguish between the given candidates.

Question: {question}
Candidates: {candidates}

Strategic Reasoning Instructions:
1. Identify the 'Core Objects' mentioned in the candidates.
2. For each object, translate it into its 'Visual Signature' (shape, color, or distribution) based on common world knowledge.
3. Focus on 'Discriminative Features': What visual details would confirm one candidate over the others?
4. Ensure the keywords cover all potential evidence needed to answer the question.

Output Requirement:
- Return ONLY a Python list of strings.
- Each keyword should be a short, visually descriptive phrase.
- DO NOT provide any conversation or explanation.

Output:
"""

GRANULARITY_TEMPLATE = """You are an expert video understanding assistant. Your task is to analyze the **Temporal Granularity** required to answer the user's question.
Output a continuous float value between 0.0 and 1.0, representing the degree of "Short-term vs. Long-term" focus.

Short Term Focus (→ 1.0):
- Focuses on immediate details, specific actions, or instantaneous events.
- Emphasizes a quick understanding of a specific scene or object.
- Can usually be answered using a single frame or frames from very closely spaced time intervals.
- Keywords: Visual attributes, object counting, OCR, momentary micro-actions.

Long Term Focus (→ 0.0):
- Focuses on overall plot development, character changes, or relationships spanning a long period.
- Emphasizes a comprehensive understanding of the video's overall content and themes.
- Requires combining frame information from multiple distant time periods.
- Keywords: Causality ("Why"), Plot summary, Evolution, Abstract themes.

Please generate the degree value of "Short-term vs. Long-term" for this query.
Question: {question}
Provide your response in the form of a float value between 0.0 and 1.0:
"""


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([rendered], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def parse_entities(content: str) -> list[str]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:python)?\s*|\s*```$", "", content, flags=re.S).strip()
    try:
        value = ast.literal_eval(content)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
    except Exception:
        pass
    items = re.split(r"\\?['\"]\s*,\s*\\?['\"]", content.strip("[] \n"))
    return [item.strip(" '\"\n") for item in items if item.strip(" '\"\n")]


def parse_granularity(content: str) -> float | str:
    match = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", content)
    if match:
        return max(0.0, min(1.0, float(match.group(0))))
    return content.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=Path("D:/ReMem/models/Qwen3-8B"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    saved = {}
    if args.output.exists():
        try:
            saved = {str(row["question_id"]): row for row in json.loads(args.output.read_text(encoding="utf-8"))}
        except Exception:
            saved = {}

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    ).eval()

    output_rows = []
    if args.limit is not None:
        records = records[: args.limit]
    for index, original in enumerate(records, 1):
        row = dict(original)
        key = str(row["question_id"])
        previous = saved.get(key, {})
        if previous.get("granularity") is not None and previous.get("entity_keywords"):
            row["granularity"] = previous["granularity"]
            row["entity_keywords"] = previous["entity_keywords"]
        else:
            if row.get("granularity") is None:
                granularity = generate(model, tokenizer, GRANULARITY_TEMPLATE.format(question=row["question"]), 16)
                row["granularity"] = parse_granularity(granularity)
            if not row.get("entity_keywords"):
                entities = generate(
                    model,
                    tokenizer,
                    ENTITY_TEMPLATE.format(question=row["question"], candidates=row["candidates"]),
                    256,
                )
                row["entity_keywords"] = parse_entities(entities)
        output_rows.append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Prepared {index}/{len(records)}: {key}", flush=True)


if __name__ == "__main__":
    main()
