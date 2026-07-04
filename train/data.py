from __future__ import annotations
 
import json
from pathlib import Path
from typing import Iterable
 
from datasets import Dataset
from transformers import AutoTokenizer
 
 
def read_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
 
 
def to_chatml(example: dict, tokenizer: AutoTokenizer) -> dict[str, str]:
    messages = example.get("messages")
    if messages is None:
        messages = [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}
 
 
def load_sft_dataset(path: str | Path, model_name: str) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    rows = [to_chatml(x, tokenizer) for x in read_jsonl(path)]
    return Dataset.from_list(rows)
