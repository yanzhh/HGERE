"""
Copy pruner prediction files and replace ner/relations with gold annotations.

For each ``ent_pred_<split>.json`` in ``pruner_dir``, looks for a matching
``<split>.jsonl`` in ``gold_dir`` and writes ``ent_pred_<split>_abbr.json``
next to the input file. Files with no matching gold split are skipped.

Usage:
    uv run gsapere-fix-gold-annos \\
        --gold_dir datasets_fixed/gsap-ere \\
        --pruner_dir saves/gsap/pruner/pelikan
"""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def process_file(pred_path: Path, gold_dir: Path) -> None:
    # ent_pred_<split>[_overlap].json → base split for gold lookup
    split = pred_path.stem.removeprefix("ent_pred_")
    base_split = split.removesuffix("_overlap")
    gold_path = gold_dir / f"{base_split}.jsonl"
    if not gold_path.exists():
        print(f"  skip {pred_path.name} — no matching gold file ({gold_path})")
        return

    out_path = pred_path.with_stem(pred_path.stem + "_abbr")
    gold_by_id = {d["doc_id"]: d for d in load_jsonl(gold_path)}
    preds = load_jsonl(pred_path)

    missing = [p["doc_id"] for p in preds if p["doc_id"] not in gold_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} doc_ids not found in {gold_path.name}: {missing[:5]}"
        )

    for pred in preds:
        gold = gold_by_id[pred["doc_id"]]
        pred["ner"] = gold["ner"]
        pred["relations"] = gold["relations"]

    with open(out_path, "w") as f:
        for pred in preds:
            f.write(json.dumps(pred) + "\n")

    print(f"  {pred_path.name} → {out_path.name}  ({len(preds)} docs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gold_dir",
        required=True,
        help="Directory with gold .jsonl files (e.g. datasets_fixed/gsap-ere)",
    )
    ap.add_argument(
        "--pruner_dir",
        required=True,
        help="Directory with pruner prediction files (ent_pred_<split>.json)",
    )
    args = ap.parse_args()

    gold_dir = Path(args.gold_dir)
    pruner_dir = Path(args.pruner_dir)

    pred_files = sorted(
        p for p in pruner_dir.glob("ent_pred_*.json") if "_abbr" not in p.stem
    )
    if not pred_files:
        raise FileNotFoundError(f"No ent_pred_*.json files found in {pruner_dir}")

    print(f"Processing {len(pred_files)} file(s) from {pruner_dir}:")
    for pred_path in pred_files:
        process_file(pred_path, gold_dir)


cli = main
