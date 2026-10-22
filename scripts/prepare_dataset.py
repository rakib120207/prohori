"""Phase 5.1 + 5.2: turn raw files into data/processed/dataset.csv.

Usage
-----
  # 1. See what's in the Mendeley file(s) you put in data/raw/
  python scripts/prepare_dataset.py --inspect

  # 2. Build the dataset (seed + anything in data/own/)
  python scripts/prepare_dataset.py

  # If label detection can't tell which value means AI:
  python scripts/prepare_dataset.py --ai-label <raw value>

Always read the sample texts it prints for each label and confirm that label 1
really is the AI-rewritten side (the spec's "check the label encoding" warning).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# Windows consoles default to a legacy codepage that cannot print Bangla
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prohori.data import CANONICAL_COLUMNS, DatasetError, normalize_table, read_table  # noqa: E402
from prohori.features import bengali_ratio  # noqa: E402


def keep_bangla_only(part: pd.DataFrame, name: str):
    """Guard: never let non-Bangla text (e.g. the English version of the dataset) into training."""
    ratios = part["text"].map(bengali_ratio)
    if ratios.median() < 0.5:
        print(f"  SKIPPING {name}: the text is not Bangla (median Bengali-letter share {ratios.median():.0%}).")
        return None
    bad = int((ratios < 0.5).sum())
    if bad:
        print(f"  dropped {bad} rows that are mostly non-Bangla")
    return part[ratios >= 0.5]

SEED_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--own-dir", default="data/own")
    ap.add_argument("--out", default="data/processed/dataset.csv")
    ap.add_argument("--inspect", action="store_true", help="only print columns / label values / samples")
    ap.add_argument("--text-col")
    ap.add_argument("--label-col")
    ap.add_argument("--pair-col")
    ap.add_argument("--source-col")
    ap.add_argument("--ai-label", help="raw label value that means AI")
    args = ap.parse_args()

    raw_files = sorted(p for p in Path(args.raw_dir).glob("*") if p.suffix.lower() in SEED_EXT)
    parts = []

    if not raw_files:
        print(f"No seed files in {args.raw_dir}/. Download the Mendeley dataset (see CONNECT.md, step 3).")
    for path in raw_files:
        print(f"\n== seed file: {path.name}")
        df = read_table(path)
        print(f"  {len(df)} rows")
        if args.inspect:
            print(df.head(3).to_string(max_colwidth=60))
        try:
            part = normalize_table(df, "seed", args.text_col, args.label_col, args.pair_col, args.source_col, args.ai_label)
            part = keep_bangla_only(part, path.name)
            if part is not None:
                parts.append(part)
        except DatasetError as e:
            print(f"\nERROR: {e}")
            if args.inspect:
                continue
            sys.exit(1)

    own_files = sorted(Path(args.own_dir).glob("*.csv"))
    for path in own_files:
        print(f"\n== own-domain file: {path.name}")
        df = read_table(path)
        if set(CANONICAL_COLUMNS) <= set(df.columns):  # already canonical (written by add_dataset.py): keep its origin tag
            part = df[CANONICAL_COLUMNS].copy()
            part["label"] = part["label"].astype(int)
            part["pair_id"] = part["pair_id"].astype(str)
            print(f"  already canonical: {len(part)} rows, origin(s) {sorted(part['origin'].astype(str).unique())}")
        else:  # written by generate_ai_rewrites.py: labels are already 0/1
            part = normalize_table(df, "own", ai_label="1", verbose=True)
        part = keep_bangla_only(part, path.name)
        if part is not None:
            parts.append(part)

    if not parts:
        sys.exit("Nothing to do.")
    data = pd.concat(parts, ignore_index=True)[CANONICAL_COLUMNS]

    print("\n== summary")
    print(data.groupby(["origin", "label"]).size().unstack(fill_value=0).rename(columns={0: "human", 1: "ai"}))
    ratio = data["label"].mean()
    print(f"AI share overall: {ratio:.1%}  (should be close to 50%)")
    for lab, name in [(0, "HUMAN (label 0)"), (1, "AI (label 1)")]:
        sample = data[data["label"] == lab].sample(min(2, (data["label"] == lab).sum()), random_state=1)
        print(f"\n-- 2 samples the pipeline will treat as {name} --")
        for t in sample["text"]:
            print("  •", t[:160].replace("\n", " "), "...")
    print("\nIf those samples look swapped, re-run with --ai-label <raw value for AI>.")

    if args.inspect:
        return
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(data)} rows)")


if __name__ == "__main__":
    main()
