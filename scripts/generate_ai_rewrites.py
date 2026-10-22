"""Phase 5.2: build your own-domain pairs.

Input : a CSV of real, consented, anonymised human Bangla texts
        columns: text            (required)
                 source          (optional, e.g. "assignment", "essay")
Output: data/own/own_pairs.csv  -> human rows (label 0) + 2-3 AI rewrites each (label 1)

  export GROQ_API_KEY=...
  python scripts/generate_ai_rewrites.py --input data/own_human_texts.csv --rewrites 2 --styles formal,casual

Keep the input file OUT of git (it is in .gitignore under data/own/*).
Groq model names change over time: check console.groq.com/docs/models and set
GROQ_MODELS="model-a,model-b" if the default is retired.

You can also add rewrites made by hand in ChatGPT / Claude / Gemini: put them
in another CSV in data/own/ with the same columns as own_pairs.csv
(text,label,pair_id,source,generator) and prepare_dataset.py will merge it.
"""
import argparse
import csv
import itertools
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Several writing styles on purpose: if every AI sample is polished and formal, a detector learns
# "polished = AI" and falsely accuses careful human writers. Casual and student-style rewrites break that shortcut.
_TAIL = "শুধু পুনর্লিখিত লেখাটি দাও, কোনো ভূমিকা বা মন্তব্য নয়।\n\n---\n{text}\n---"
STYLES = {
    "formal": "নিচের লেখাটি তোমার নিজের ভাষায় নতুন করে লেখো। মূল অর্থ ও দৈর্ঘ্য প্রায় একই রাখো। " + _TAIL,
    "casual": "নিচের লেখাটি এমনভাবে নতুন করে লেখো যেন একজন সাধারণ মানুষ দৈনন্দিন কথ্য ভঙ্গিতে লিখছে: সহজ শব্দ, ছোট বাক্য। "
              "মূল অর্থ ও দৈর্ঘ্য প্রায় একই রাখো। " + _TAIL,
    "student": "একজন স্কুল বা কলেজের শিক্ষার্থী যেভাবে লেখে সেভাবে নিচের লেখাটি নতুন করে লেখো: সাধারণ শব্দ, খুব পরিপাটি বা সাহিত্যিক নয়। "
               "মূল অর্থ ও দৈর্ঘ্য প্রায় একই রাখো। " + _TAIL,
}


def rewrite(client, model, key, text, style="formal", retries=5):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": STYLES[style].format(text=text)}],
        "temperature": 0.9,
    }
    for attempt in range(retries):
        r = client.post(API_URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=120)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt * 2)
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("Groq kept rate-limiting / erroring; try again later or use fewer rewrites.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/own/own_pairs.csv")
    ap.add_argument("--rewrites", type=int, default=2, help="AI rewrites per human text (2-3 recommended)")
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between calls")
    ap.add_argument("--styles", default="formal,casual", help="comma list from: " + ",".join(STYLES) + " (rotated across rewrites)")
    args = ap.parse_args()

    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit("Set GROQ_API_KEY first (see .env.example).")
    models = [m.strip() for m in os.environ.get("GROQ_MODELS", "llama-3.3-70b-versatile").split(",") if m.strip()]
    humans = pd.read_csv(args.input, encoding="utf-8-sig")
    if "text" not in humans.columns:
        sys.exit("Input CSV needs a 'text' column.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_pairs = set()
    new_file = not out_path.exists()
    if not new_file:
        done_pairs = set(pd.read_csv(out_path)["pair_id"].astype(str))

    styles = [x.strip() for x in args.styles.split(",") if x.strip()]
    bad = [x for x in styles if x not in STYLES]
    if bad:
        sys.exit(f"Unknown style(s) {bad}. Choose from {list(STYLES)}")
    model_cycle = itertools.cycle(models)
    style_cycle = itertools.cycle(styles)
    with open(out_path, "a", newline="", encoding="utf-8-sig") as f, httpx.Client() as client:
        w = csv.DictWriter(f, fieldnames=["text", "label", "pair_id", "source", "generator", "style", "domain"])
        if new_file:
            w.writeheader()
        for i, row in humans.iterrows():
            pid = f"h{i}"
            if pid in done_pairs:
                continue
            src = row.get("source", "assignment") if "source" in humans.columns else "assignment"
            w.writerow({"text": row["text"], "label": 0, "pair_id": pid, "source": src, "generator": "human", "style": "human", "domain": src})
            for _ in range(args.rewrites):
                model = next(model_cycle)
                style = next(style_cycle)
                try:
                    ai = rewrite(client, model, key, row["text"], style)
                except Exception as e:  # keep going; one bad call shouldn't lose the run
                    print(f"  ! row {i} ({model}): {e}")
                    continue
                # AI rows carry the GENERATOR as their source (like the seed dataset), the domain goes in its own column
                w.writerow({"text": ai, "label": 1, "pair_id": pid, "source": model, "generator": model, "style": style, "domain": src})
                time.sleep(args.sleep)
            f.flush()
            print(f"row {i + 1}/{len(humans)} done")
    print(f"\nwrote {out_path}. Next: python scripts/prepare_dataset.py && python scripts/train.py")


if __name__ == "__main__":
    main()
