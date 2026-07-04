"""Generate predictions from the local API for every row in eval_set.jsonl."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib import request


def main(
    eval_set: str = "eval/eval_set.jsonl",
    out: str = "eval/predictions.jsonl",
    endpoint: str = "http://localhost:8000",
    max_new_tokens: int = 80,
) -> None:
    rows = [json.loads(l) for l in Path(eval_set).read_text().splitlines() if l.strip()]
    results = []
    for row in rows:
        payload = json.dumps({"prompt": row["input"], "max_new_tokens": max_new_tokens, "stream": False}).encode("utf-8")
        req = request.Request(
            f"{endpoint}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        prediction = str(body.get("text", "")).strip()
        results.append(
            {
                "id": row["id"],
                "input": row["input"],
                "prediction": prediction,
                "reference": row["reference"],
                "schema": row.get("schema"),
                "task": row.get("task"),
            }
        )
        print(f"[{row['id']}] {prediction[:80]!r}")

    Path(out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results))
    print(f"\nWrote {len(results)} predictions to {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
