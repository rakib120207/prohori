"""Robust, honest evaluation -> models/cv_report.json (shown in the app's "How accurate is this?" panel).

  python scripts/evaluate_cv.py --exclude-sources wikisource

What it measures (takes about 5-8 minutes):
  * pooled 5-fold CV where human/AI twins never straddle train and test
  * accuracy and false-flag rate BY TEXT LENGTH
  * selective accuracy: how right the app is when it only answers if it is confident
  * leave-one-generator-out: how well it catches an LLM it has never seen
  * a SYNTHETIC long-text simulation (whole-block vs chunked scoring)
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
from prohori.model import cross_validate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/dataset.csv")
    ap.add_argument("--out", default="models/cv_report.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--exclude-sources", nargs="*", default=[])
    args = ap.parse_args()

    rep = cross_validate(load_processed(args.data), n_splits=args.folds, exclude_sources=args.exclude_sources)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    p = rep["pooled"]
    print(f"\n=== pooled {rep['folds']}-fold CV: {rep['n_rows']} texts / {rep['n_pairs']} pairs ===")
    print(f"accuracy {p['accuracy']:.3f}   ROC-AUC {p['roc_auc']:.3f}   AI precision {p['ai_precision']:.3f}   AI recall {p['ai_recall']:.3f}")
    print("human false-flag rate at thresholds:", json.dumps(p["human_false_positive_rate_at"]))
    print("\nby text length:")
    for k, v in p["by_length"].items():
        print(f"  {k:16} n={v['n']:4} accuracy={v['accuracy']:.3f} human false-flag={v['human_false_flag_rate']}")
    print("\nselective (only answer when confident):")
    for r in p["selective"]:
        print(f"  confidence >= {r['min_confidence']:.1f}: covers {r['coverage']:.0%}, right {r['accuracy_when_answering']:.1%}")
    print("\nunseen generator (leave-one-out):")
    for g, v in rep["unseen_generator"].items():
        print(f"  {g:10} caught unseen {v['ai_caught_when_unseen']:.2f} vs seen {v['ai_caught_when_seen_in_training']:.2f}  (human twins correct {v['human_correct_on_twins']:.2f})")
    print("\nlong-text simulation (SYNTHETIC):")
    for k, v in rep["long_text_simulation"].items():
        if k == "note":
            continue
        print(f"  ~{v['avg_words']} words: whole-block {v['whole_block']}  chunked {v['chunked']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
