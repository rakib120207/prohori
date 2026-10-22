"""Phase 5.9 (OPTIONAL v2): fine-tune csebuetnlp/banglabert_small as a binary classifier.

Run this on a free Colab / Kaggle T4 GPU, NOT on your local AMD machine. It evaluates
on EXACTLY the same held-out pairs as scripts/train.py (same group split + seed), so the
two numbers are directly comparable.

Colab cells:
    !pip install -q transformers datasets accelerate scikit-learn pandas
    !pip install -q git+https://github.com/csebuetnlp/normalizer
    # upload data/processed/dataset.csv (and models/meta.json if you want the side-by-side)
    !python finetune_banglabert.py --data dataset.csv --meta meta.json

NOT executed in the build sandbox (no GPU / no HuggingFace access there), so treat it as
reviewed-but-unrun code: if a library version changed an argument name, the error will
point at the line.

This script only EVALUATES the transformer. It is intentionally not wired into the live
API: the spec's design is that scoring uses the classical model, and you should only
adopt the transformer if it clearly beats it on the same held-out set.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

MODEL_NAME = "csebuetnlp/banglabert_small"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.csv")
    ap.add_argument("--meta", default="meta.json")
    ap.add_argument("--out", default="banglabert_ai_detector")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-sources", nargs="*", default=["wikisource"],
                    help="drop pairs whose HUMAN text came from these sources (must match train.py)")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from normalizer import normalize
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    df = pd.read_csv(args.data).reset_index(drop=True)
    if args.exclude_sources:  # same pair-level exclusion as prohori.model.exclude_pairs_by_source
        want = {x.lower() for x in args.exclude_sources}
        hs = df[df["label"] == 0].set_index("pair_id")["source"].astype(str).str.lower()
        df = df[~df["pair_id"].isin(set(hs[hs.isin(want)].index))].reset_index(drop=True)
        print(f"after excluding {sorted(want)}: {len(df)} rows")
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    tr_idx, te_idx = next(splitter.split(df, df["label"], groups=df["pair_id"]))  # same as prohori.model.group_split
    train_df, test_df = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
    for part in (train_df, test_df):
        part["text"] = part["text"].map(normalize)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def encode(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    to_ds = lambda d: Dataset.from_pandas(d[["text", "label"]].reset_index(drop=True)).map(encode, batched=True)
    train_ds, test_ds = to_ds(train_df), to_ds(test_df)

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=test_ds,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics)
    trainer.train()

    out = trainer.predict(test_ds)
    logits = out.predictions
    proba = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    pred = logits.argmax(-1)
    y = out.label_ids
    print("\n=========== BanglaBERT (held-out, same pairs as classical model) ===========")
    print(classification_report(y, pred, target_names=["human", "ai"], zero_division=0))
    auc = roc_auc_score(y, proba[:, 1])
    print(f"accuracy {accuracy_score(y, pred):.3f}   ROC-AUC {auc:.3f}")

    if Path(args.meta).exists():
        m = json.loads(Path(args.meta).read_text(encoding="utf-8"))["metrics"]
        print(f"\nclassical XGBoost on the same held-out set: accuracy {m['accuracy']:.3f}   ROC-AUC {m['roc_auc']}")
        print("Adopt the transformer only if the gap is large enough to justify its cost and size.")

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
