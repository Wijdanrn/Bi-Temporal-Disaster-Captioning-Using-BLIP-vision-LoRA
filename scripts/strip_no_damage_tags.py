"""
Fase 1 cleanup -- strip leaked <NO_DAMAGE>/</NO_DAMAGE> annotation-template
tags from ground_truth_en and ground_truth_id. Confirmed present in the
original English source (not introduced by translation) via
docs/design_decisions.md Fase 3 risk #3 -- these are DisasterM3 template
artifacts, not translation errors, but they'd otherwise be learned and
emitted verbatim by the model.

Usage:
    python scripts/strip_no_damage_tags.py --split train
    python scripts/strip_no_damage_tags.py --split test
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TAG_RE = re.compile(r"\s*<\s*/?\s*NO_DAMAGE\s*>\s*")


def clean(text: str) -> str:
    text = TAG_RE.sub(" ", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r"\n +", "\n", text)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    args = ap.parse_args()

    path = ROOT / "data" / "interim" / f"{args.split}_id.jsonl"
    records = [json.loads(l) for l in open(path, encoding="utf-8")]

    n_changed = 0
    for r in records:
        new_en = clean(r["ground_truth_en"])
        new_id = clean(r["ground_truth_id"])
        if new_en != r["ground_truth_en"] or new_id != r["ground_truth_id"]:
            n_changed += 1
        r["ground_truth_en"] = new_en
        r["ground_truth_id"] = new_id
        for section_dict_key in ("sections_en", "sections_id"):
            if section_dict_key in r:
                r[section_dict_key] = {k: clean(v) for k, v in r[section_dict_key].items()}

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[strip] {args.split}: {n_changed}/{len(records)} records changed. Overwrote {path}")


if __name__ == "__main__":
    main()
