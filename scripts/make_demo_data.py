"""SMOKE-TEST DATA ONLY. Generates fake Bangla-looking paired text so you can
verify the whole pipeline runs before you download the real dataset.

The "AI" side is given smoother sentence lengths and a heavier function-word
share; the "human" side is messier. Any accuracy you get from this data says
NOTHING about real-world performance.

  python scripts/make_demo_data.py            # writes data/raw/demo_seed.csv
"""
import argparse
import random
from pathlib import Path

import pandas as pd

CONTENT = (
    "বাংলাদেশ নদী গ্রাম শহর মানুষ শিক্ষা বই স্কুল কলেজ বিশ্ববিদ্যালয় ছাত্র শিক্ষক পরীক্ষা ফল "
    "কৃষক ধান মাঠ বৃষ্টি বন্যা বাজার দাম খবর সরকার উন্নয়ন প্রযুক্তি কম্পিউটার ইন্টারনেট ভাষা সাহিত্য কবিতা "
    "গল্প ইতিহাস সংস্কৃতি উৎসব খেলা ক্রিকেট ফুটবল স্বাস্থ্য হাসপাতাল ডাক্তার ওষুধ পরিবেশ গাছ বন পাখি "
    "আকাশ সূর্য চাঁদ সমুদ্র পাহাড় যাত্রা রাস্তা গাড়ি রেল নৌকা শ্রম কারখানা পোশাক রপ্তানি অর্থনীতি ব্যাংক"
).split()
FUNCTION = "এই সেই এবং কিন্তু তাই যে কি না হয় ছিল আছে করে হবে থেকে জন্য সাথে মধ্যে খুব আরও তবে যদি তখন".split()


def sentence(rng, n_words, func_share):
    words = []
    for _ in range(n_words):
        words.append(rng.choice(FUNCTION) if rng.random() < func_share else rng.choice(CONTENT))
    mark = rng.choices(["।", "?", "!"], weights=[0.92, 0.05, 0.03])[0]
    return " ".join(words) + mark


def human_text(rng):
    out = []
    for _ in range(rng.randint(5, 10)):
        out.append(sentence(rng, rng.choice([4, 6, 9, 14, 22, 31]) + rng.randint(-2, 3), 0.28))
    return " ".join(out)


def ai_twin(rng, human):
    n = len(human.split()) // 8 or 6
    out = []
    for _ in range(max(5, n)):
        out.append(sentence(rng, rng.randint(11, 15), 0.38))
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--out", default="data/raw/demo_seed.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for i in range(args.pairs):
        h = human_text(rng)
        rows.append({"Pair_ID": i, "Text": h, "Label": "Human", "Source": "demo"})
        rows.append({"Pair_ID": i, "Text": ai_twin(rng, h), "Label": "AI", "Source": "demo"})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(rows)} rows) - FAKE data, pipeline smoke test only")


if __name__ == "__main__":
    main()
