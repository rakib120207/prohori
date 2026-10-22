"""Turn a raw HUMAN corpus into passages ready for AI rewriting.

  python scripts/import_corpus.py --input "D:\\downloads\\potrika" --domain news_potrika --n 300
  python scripts/import_corpus.py --input bnwiki_articles.csv --text-col text --domain wikipedia --n 300

Accepts a file (csv / tsv / xlsx / json / jsonl / txt) or a folder (every .txt / .csv / .xlsx inside, recursively).
A .txt file is one document (blank lines separate paragraphs). Output: data/own_human_texts.csv with columns
    text, source        (source = the domain name you chose; it is what the app reports per domain)

Rules the script enforces so the data stays useful:
  * keeps only mostly-Bangla text (default 85% Bengali letters)
  * cuts long articles into passages of 40-150 words (the length the model is trained and scored on)
  * drops exact duplicates, and appends to an existing output file
HUMAN-ONLY RULE: use only text published BEFORE 2022 (before LLM text flooded the web), or your own consented
writing. Otherwise some of your "human" samples are secretly AI and the model learns the wrong thing.
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prohori.bntok import NLTKTokenizer  # noqa: E402
from prohori.data import read_table  # noqa: E402
from prohori.features import bengali_ratio  # noqa: E402

TEXT_COL_GUESS = ["text", "content", "article", "news", "body", "sample", "passage", "paragraph"]
tok = NLTKTokenizer()


def iter_documents(path: Path, text_col):
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix.lower() in {".txt", ".csv", ".tsv", ".xlsx", ".json", ".jsonl"})
    for f in files:
        if f.suffix.lower() == ".txt":
            yield f.read_text(encoding="utf-8", errors="ignore")
            continue
        df = read_table(f)
        col = text_col or next((c for c in df.columns if str(c).strip().lower() in TEXT_COL_GUESS), None)
        if col is None:
            col = max((c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")),
                      key=lambda c: df[c].astype(str).str.len().mean())
            print(f"  {f.name}: no obvious text column, using the longest-text column {col!r}")
        yield from df[col].dropna().astype(str)


def passages(doc: str, rng: random.Random, lo: int, hi: int, per_doc: int):
    sents = [s.strip() for s in tok.sentence_tokenize(doc) if s.strip()]
    out = []
    for _ in range(per_doc):
        if not sents:
            break
        start = rng.randrange(len(sents))
        target = rng.randint(lo + 10, hi - 10)
        cur, n = [], 0
        for s in sents[start:]:
            cur.append(s)
            n += len(s.split())
            if n >= target:
                break
        text = " ".join(cur)
        if lo <= len(text.split()) <= hi:
            out.append(text)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--domain", required=True, help="short name, e.g. news_potrika, wikipedia, social, student_essay")
    ap.add_argument("--out", default="data/own_human_texts.csv")
    ap.add_argument("--text-col")
    ap.add_argument("--n", type=int, default=300, help="passages to keep from this corpus")
    ap.add_argument("--min-words", type=int, default=40)
    ap.add_argument("--max-words", type=int, default=150)
    ap.add_argument("--per-doc", type=int, default=1, help="passages cut from each article")
    ap.add_argument("--min-bengali", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cand, seen_docs = [], 0
    for doc in iter_documents(Path(args.input), args.text_col):
        seen_docs += 1
        if seen_docs > 200_000:
            break
        if bengali_ratio(doc) < args.min_bengali:
            continue
        cand += passages(doc, rng, args.min_words, args.max_words, args.per_doc)
    print(f"read {seen_docs} documents -> {len(cand)} candidate passages")
    cand = [t for t in dict.fromkeys(cand) if bengali_ratio(t) >= args.min_bengali]
    rng.shuffle(cand)
    cand = cand[: args.n]
    if not cand:
        sys.exit("No usable passages. Check --text-col and that the text is Bangla.")

    new = pd.DataFrame({"text": cand, "source": args.domain})
    out = Path(args.out)
    if out.exists():
        old = pd.read_csv(out, encoding="utf-8-sig")
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(subset="text")
    out.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"wrote {out}: {len(new)} human passages total")
    print(new.groupby("source").size().to_string())
    print("\nsample:\n  " + cand[0][:200].replace("\n", " "))
    print("\nNext: python scripts/generate_ai_rewrites.py --input", out, "--rewrites 2 --styles formal,casual")


if __name__ == "__main__":
    main()
