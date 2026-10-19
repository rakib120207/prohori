"""Phase 5.8: blind test on fresh texts from real people.

Make data/blind/ with a CSV (default data/blind/blind_texts.csv):
    text, truth          truth = "human" or "ai"   (kept secret from the tool)

  python scripts/blind_test.py
  python scripts/blind_test.py --file data/blind/blind_texts.csv --threshold 0.65

Writes data/blind/blind_results.csv so you can read the misses and note where it got confused.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

for _stream in (sys.stdout, sys.stderr):  # Windows consoles can't print Bangla by default
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prohori.model import Predictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/blind/blind_texts.csv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.file, encoding="utf-8-sig")
    if not {"text", "truth"} <= set(df.columns):
        sys.exit("CSV needs columns: text, truth (human|ai)")
    pred = Predictor(args.model_dir)
    rows = []
    for _, r in df.iterrows():
        try:
            out = pred.analyze(r["text"])
        except ValueError as e:
            print("skipped:", e)
            continue
        p = out["ai_probability"]
        rows.append(
            {
                "truth": str(r["truth"]).lower(),
                "ai_probability": p,
                "predicted": "ai" if p >= args.threshold else "human",
                "top_reason": out["top_reasons"][0],
                "words": out["stats"]["words"],
                "text": r["text"][:80].replace("\n", " "),
            }
        )
    res = pd.DataFrame(rows)
    res["correct"] = res["truth"] == res["predicted"]
    Path(args.file).with_name("blind_results.csv").write_text(res.to_csv(index=False), encoding="utf-8-sig")

    n = len(res)
    print(f"\nblind hit rate: {res['correct'].sum()}/{n} = {res['correct'].mean():.1%}  (threshold {args.threshold})")
    h = res[res["truth"] == "human"]
    a = res[res["truth"] == "ai"]
    if len(h):
        print(f"humans wrongly flagged (false positives): {(h['predicted'] == 'ai').sum()}/{len(h)}")
    if len(a):
        print(f"AI texts missed: {(a['predicted'] == 'human').sum()}/{len(a)}")
    print("\nmisses:")
    print(res[~res["correct"]][["truth", "ai_probability", "top_reason", "words", "text"]].to_string(index=False))


if __name__ == "__main__":
    main()
