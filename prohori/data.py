"""Dataset loading / normalization (spec Sections 5.1 and 5.2).

The Mendeley seed dataset's exact column names and label encoding could not be
verified offline, so this module detects them and PRINTS what it decided. The
spec's warning applies: confirm the label direction yourself (prepare_dataset.py
shows sample texts per label so you can eyeball it).

Canonical schema written to data/processed/dataset.csv:
    text, label (0 = human, 1 = AI), pair_id, source, origin ("seed" | "own")
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

CANONICAL_COLUMNS = ["text", "label", "pair_id", "source", "origin"]

_TEXT_NAMES = ["text", "content", "sample", "sample_text", "article", "passage", "body", "paragraph"]
_LABEL_NAMES = ["label", "class", "target", "type", "category_label", "is_ai", "ai_generated", "y"]
_PAIR_HINTS = ["pair"]
_SOURCE_HINTS = ["source", "domain", "genre", "category"]


class DatasetError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".tsv"}:
        return pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None)  # all sheets
        if len(sheets) == 1:
            return next(iter(sheets.values()))
        print(f"  workbook has {len(sheets)} sheets: { {k: len(v) for k, v in sheets.items()} }")
        frames = list(sheets.values())
        if all(list(f.columns) == list(frames[0].columns) for f in frames):
            print("  sheets share the same columns -> stacking them")
            return pd.concat(frames, ignore_index=True)
        biggest = max(sheets, key=lambda k: len(sheets[k]))
        print(f"  sheets differ in layout -> using the largest sheet {biggest!r}. Split the workbook if that's wrong.")
        return sheets[biggest]
    if suffix in {".json"}:
        return pd.read_json(path)
    if suffix in {".jsonl"}:
        return pd.read_json(path, lines=True)
    raise DatasetError(f"Unsupported file type: {path.name}")


def _find(columns, names=None, hints=None):
    lowered = {c: str(c).strip().lower() for c in columns}
    if names:
        for want in names:
            for col, low in lowered.items():
                if low == want:
                    return col
    if hints:
        for col, low in lowered.items():
            if any(h in low for h in hints):
                return col
    return None


# --------------------------------------------------------------------------- #
# Label normalization
# --------------------------------------------------------------------------- #
_AI_WORDS = re.compile(r"\b(ai|a\.i\.|llm|machine|generated|rewritten|synthetic|chatgpt|gpt|claude|gemini|deepseek|bot)\b", re.I)
_HUMAN_WORDS = re.compile(r"\b(human|person|original|authentic|real|natural)\b", re.I)


def normalize_label(value, ai_label=None) -> int | None:
    """Map a raw label cell to 0 (human) / 1 (AI). Returns None if unknown.

    `ai_label` lets you state explicitly which RAW value means AI, e.g. "1",
    "AI", "ai_rewritten". Everything else is then human.
    """
    if pd.isna(value):
        return None
    raw = str(value).strip()
    if ai_label is not None:
        return 1 if raw.lower() == str(ai_label).strip().lower() else 0
    low = raw.lower()
    if low in {"0", "0.0", "false", "no"}:
        return 0
    if low in {"1", "1.0", "true", "yes"}:
        return 1
    # text label: "human"/"ai" etc. "human" wins so "Human-written" is never read as AI
    if _HUMAN_WORDS.search(low) and not _AI_WORDS.search(low):
        return 0
    if _AI_WORDS.search(low):
        return 1
    return None


# --------------------------------------------------------------------------- #
# Normalizing a raw table to the canonical schema
# --------------------------------------------------------------------------- #
def normalize_table(
    df: pd.DataFrame,
    origin: str,
    text_col=None,
    label_col=None,
    pair_col=None,
    source_col=None,
    ai_label=None,
    verbose: bool = True,
) -> pd.DataFrame:
    cols = list(df.columns)
    log = print if verbose else (lambda *a, **k: None)
    log(f"  columns found: {cols}")

    text_col = text_col or _find(cols, _TEXT_NAMES, hints=["text"])
    label_col = label_col or _find(cols, _LABEL_NAMES)
    pair_col = pair_col or _find(cols, hints=_PAIR_HINTS)
    source_col = source_col or _find(cols, hints=_SOURCE_HINTS)

    # Wide format fallback: one column of human text, one of AI text per row.
    if label_col is None:
        human_cols = [c for c in cols if "human" in str(c).lower()]
        ai_cols = [c for c in cols if re.search(r"\b(ai|rewrit|generat|llm)", str(c).lower())]
        if human_cols and ai_cols:
            log(f"  wide format detected: human={human_cols[0]!r} ai={ai_cols[0]!r}")
            base = df.reset_index().rename(columns={"index": "_row"})
            human = pd.DataFrame({"text": base[human_cols[0]], "label": 0, "pair_id": base["_row"]})
            ai = pd.DataFrame({"text": base[ai_cols[0]], "label": 1, "pair_id": base["_row"]})
            out = pd.concat([human, ai], ignore_index=True)
            out["source"] = ""
            return _finalize(out, origin, log)

    if text_col is None:
        # last resort: the column with the longest average string length
        str_cols = [c for c in cols if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object]
        if not str_cols:
            raise DatasetError("Could not find a text column. Pass --text-col.")
        text_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
        log(f"  no obvious text column; guessing longest-text column {text_col!r}")
    if label_col is None:
        raise DatasetError(f"Could not find a label column in {cols}. Pass --label-col.")

    log(f"  using text={text_col!r} label={label_col!r} pair={pair_col!r} source={source_col!r}")
    raw_counts = df[label_col].value_counts(dropna=False).to_dict()
    log(f"  raw label values: {raw_counts}")

    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "label": df[label_col].map(lambda v: normalize_label(v, ai_label)),
            "pair_id": df[pair_col].astype(str) if pair_col else None,
            "source": df[source_col].astype(str) if source_col else "",
        }
    )
    unknown = out["label"].isna().sum()
    if unknown:
        raise DatasetError(
            f"{unknown} rows have a label I can't interpret. Raw values: {raw_counts}. "
            "Re-run with --ai-label <the raw value that means AI>."
        )
    return _finalize(out, origin, log)


def _finalize(out: pd.DataFrame, origin: str, log) -> pd.DataFrame:
    out = out.copy()
    out["label"] = out["label"].astype(int)
    out["text"] = out["text"].astype(str).str.strip()
    before = len(out)
    out = out[out["text"].str.split().str.len() >= 5]  # drop empties / fragments
    out = out.drop_duplicates(subset=["text", "label"])
    if len(out) != before:
        log(f"  dropped {before - len(out)} empty/short/duplicate rows")
    if "pair_id" not in out or out["pair_id"].isna().all():
        log("  WARNING: no pair id found - train/test split cannot keep human/AI twins together.")
        out["pair_id"] = [f"{origin}-row{i}" for i in range(len(out))]
    else:
        out["pair_id"] = origin + "-" + out["pair_id"].astype(str)
    out["origin"] = origin
    return out[CANONICAL_COLUMNS].reset_index(drop=True)


def load_processed(path: str | Path = "data/processed/dataset.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"{path} not found. Run scripts/prepare_dataset.py first.")
    df = pd.read_csv(path)
    missing = set(CANONICAL_COLUMNS) - set(df.columns)
    if missing:
        raise DatasetError(f"{path} is missing columns: {missing}")
    return df
