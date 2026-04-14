"""
Copy a pruner prediction file and replace ner/relations with gold annotations.

The output is written next to the input file with an ``_abbr`` suffix, e.g.
``ent_pred_dev.json`` → ``ent_pred_dev_abbr.json``.

Usage:
    uv run gsapere-fix-gold-annos \\
        --fixed_gold_path datasets_fixed/gsap-ere/dev.jsonl \\
        --pruner_path saves/gsap/pruner/pelikan/ent_pred_dev.json
"""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fixed_gold_path",
        required=True,
        help="Path to the fixed gold .jsonl file (datasets_fixed/.../<split>.jsonl)",
    )
    ap.add_argument(
        "--pruner_path",
        required=True,
        help="Path to the pruner prediction file (ent_pred_<split>.json)",
    )
    args = ap.parse_args()

    gold_path = Path(args.fixed_gold_path)
    pruner_path = Path(args.pruner_path)
    out_path = pruner_path.with_stem(pruner_path.stem + "_abbr")

    gold_by_id = {d["doc_id"]: d for d in load_jsonl(gold_path)}
    preds = load_jsonl(pruner_path)

    missing = [p["doc_id"] for p in preds if p["doc_id"] not in gold_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} doc_ids not found in gold file: {missing[:5]}"
        )

    for pred in preds:
        gold = gold_by_id[pred["doc_id"]]
        pred["ner"] = gold["ner"]
        pred["relations"] = gold["relations"]

    with open(out_path, "w") as f:
        for pred in preds:
            f.write(json.dumps(pred) + "\n")

    print(f"Wrote {len(preds)} docs to {out_path}")


cli = main
