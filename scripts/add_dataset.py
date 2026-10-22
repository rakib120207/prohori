"""Convert ANY labeled Human-vs-AI Bangla file into the project's format and drop it in data/own/.

  python scripts/add_dataset.py --file data/external/tvtm9tkvdn.xlsx --name mendeley_6k ^
         --text-col source_text --label-col Source --ai-label AI --source-col "Source Name" --preview

Run with --preview first: it prints the detected columns, label counts and 2 sample texts per label so you can
confirm the label direction. Then run again without --preview. The result is data/own/<name>.csv, which
prepare_dataset.py merges automatically (origin tag = <name>, so results are reported per dataset).

To only TEST on a dataset without training on it, use scripts/eval_external.py instead.
"""
import argparse
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prohori.data import normalize_table, read_table  # noqa: E402
from prohori.features import bengali_ratio  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--name", required=True, help="short tag, e.g. ieee_social, mendeley_6k")
    ap.add_argument("--text-col")
    ap.add_argument("--label-col")
    ap.add_argument("--ai-label", help="raw label value that means AI (e.g. AI, 1, GPT)")
    ap.add_argument("--pair-col")
    ap.add_argument("--source-col")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    df = read_table(args.file)
    print(f"{len(df)} rows")
    part = normalize_table(df, args.name, args.text_col, args.label_col, args.pair_col, args.source_col, args.ai_label)
    ratios = part["text"].map(bengali_ratio)
    if ratios.median() < 0.5:
        sys.exit(f"This file is not Bangla (median Bengali share {ratios.median():.0%}). Nothing written.")
    part = part[ratios >= 0.5]
    print(part.groupby("label").size().rename({0: "human", 1: "ai"}).to_string())
    for lab, nm in [(0, "HUMAN"), (1, "AI")]:
        for t in part[part.label == lab].text.head(2):
            print(f"  [{nm}] {t[:140]}")
    if args.preview:
        print("\n(preview only; nothing written). If the HUMAN/AI samples look swapped, set --ai-label.")
        return
    out = Path("data/own") / f"{args.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    part.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\nwrote {out}. Next: python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
