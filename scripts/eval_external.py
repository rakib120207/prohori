"""Test the trained model on a dataset it has NEVER seen (no training, no tuning). The most honest generalization check.

  python scripts/eval_external.py --file data/external/tvtm9tkvdn.xlsx --text-col source_text ^
         --label-col Source --ai-label AI --source-col "Source Name"

Reports accuracy, AUC, how often real human text is flagged, how often AI is caught, and a breakdown per source /
generator and per text length. Run it on every new dataset BEFORE you decide to train on it.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.metrics import roc_auc_score  # noqa: E402

from prohori.data import normalize_table, read_table  # noqa: E402
from prohori.features import bengali_ratio  # noqa: E402
from prohori.model import Predictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--text-col")
    ap.add_argument("--label-col")
    ap.add_argument("--ai-label")
    ap.add_argument("--pair-col")
    ap.add_argument("--source-col")
    ap.add_argument("--max", type=int, default=2000, help="score at most this many rows (random sample)")
    args = ap.parse_args()

    df = normalize_table(read_table(args.file), "external", args.text_col, args.label_col, args.pair_col, args.source_col, args.ai_label)
    df = df[df["text"].map(bengali_ratio) >= 0.5]
    if len(df) > args.max:
        df = df.sample(args.max, random_state=1)
    df = df.reset_index(drop=True)
    pred = Predictor(args.model_dir)
    p = np.array([pred.scorer.score_text(t).probability for t in df["text"]])
    y = df["label"].to_numpy()
    hi, lo = pred.bands["high"], pred.bands["low"]
    hum, ai = y == 0, y == 1
    answered = (p >= hi) | (p <= lo)

    print(f"\n=== external test: {len(df)} texts ({hum.sum()} human / {ai.sum()} AI), never seen in training ===")
    print(f"accuracy @0.5: {((p >= .5) == y).mean():.3f}   ROC-AUC: {roc_auc_score(y, p):.3f}")
    print(f"real human text flagged (p>={hi}): {(p[hum] >= hi).mean():.1%}   AI caught (p>={hi}): {(p[ai] >= hi).mean():.1%}")
    print(f"answers (not 'inconclusive'): {answered.mean():.0%} of texts; right when it answers: {(((p >= .5) == y)[answered]).mean():.1%}")
    words = df["text"].str.split().str.len()
    print("\nby text length:")
    for lo_w, hi_w in [(0, 40), (40, 60), (60, 90), (90, 10**6)]:
        m = ((words > lo_w) & (words <= hi_w)).to_numpy()
        if m.sum() >= 10:
            print(f"  {lo_w}-{hi_w if hi_w < 10**6 else '+'} words  n={m.sum():4}  accuracy {((p[m] >= .5) == y[m]).mean():.3f}")
    print("\nby source (human: share left alone; ai: share caught at 0.5):")
    d = df.assign(p=p, src=df["source"].astype(str).str.lower())
    for (lab, src), g in d.groupby(["label", "src"]):
        ok = (g.p < .5).mean() if lab == 0 else (g.p >= .5).mean()
        print(f"  {'human' if lab == 0 else 'ai   '} {src[:28]:28} n={len(g):4}  {ok:.3f}")


if __name__ == "__main__":
    main()
