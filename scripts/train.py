"""Phase 5.4: train, evaluate honestly, save to models/.

  python scripts/train.py                 # train on 80%, report on the held-out 20%
  python scripts/train.py --refit-all     # same report, but ship a model refit on 100% of data
"""
import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):  # Windows consoles can't print Bangla by default
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prohori.data import load_processed  # noqa: E402
from prohori.model import save, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/dataset.csv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refit-all", action="store_true")
    ap.add_argument("--exclude-sources", nargs="*", default=[],
                    help="drop pairs whose HUMAN text came from these sources, e.g. wikisource")
    args = ap.parse_args()

    df = load_processed(args.data)
    print(f"loaded {len(df)} rows; AI share {df['label'].mean():.1%}")
    result = train(df, test_size=args.test_size, seed=args.seed, refit_all=args.refit_all,
                   exclude_sources=args.exclude_sources)
    m = result.meta["metrics"]

    print("\n================ HELD-OUT RESULTS (never seen in training) ================")
    r = m["report"]
    print(f"{'':8}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for k in ("human", "ai"):
        print(f"{k:8}{r[k]['precision']:>10.3f}{r[k]['recall']:>10.3f}{r[k]['f1-score']:>10.3f}{int(r[k]['support']):>10}")
    print(f"\naccuracy: {m['accuracy']:.3f}   ROC-AUC: {m['roc_auc']}")
    print(f"confusion matrix [[TN FP],[FN TP]]: {m['confusion_matrix']['values']}")
    print(f"AI-class precision: {m['ai_precision']:.3f}   <- watch this: 1 - precision is how often a 'flag' is wrong")
    print(f"human false-positive rate at thresholds: {json.dumps(m['human_false_positive_rate_at'])}")
    print(f"accuracy by origin: {json.dumps(m['accuracy_by_origin'])}")
    print("\nper-source correct rate (human: how often real writing is left alone; ai: how often that generator is caught):")
    for k, v in sorted(m["correct_rate_by_source"].items()):
        print(f"  {k:28} n={v['n']:4}  {v['correct_rate']:.3f}")
    save(result, args.model_dir)
    print(f"\nsaved model.joblib, lm.joblib, surface.joblib, meta.json to {args.model_dir}/")


if __name__ == "__main__":
    main()
