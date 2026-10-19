"""Training + inference for Prohori (spec Sections 5.4 and 5.5).

Deliberate deviations from the spec's sample code (all explained in CONNECT.md):

1. Split by Pair_ID, not by row. The seed dataset pairs each human text with its
   AI rewrite; a row-level split leaks one twin into the other side.
2. Cross-fit every learned feature. The perplexity LM and the surface model are
   always scored out-of-fold for training rows (never on texts they were fit on).
3. Extra explainable stylometric features + a character-pattern "surface" score,
   because the spec's six features alone reached only ~62% on the real data.
4. --exclude-sources, to drop a confounded sub-corpus (WikiSource's archaic
   spelling lets a model "detect human" by spelling era instead of by writing).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .features import (
    FEATURE_NAMES,
    Doc,
    base_features,
    bengali_ratio,
    doc_stats,
    features_from_doc,
    perplexity,
    tokenize,
    train_perplexity_model_from_docs,
)
from .surface import SurfaceModel

MODEL_VERSION = "prohori-v1.1"
SURFACE_FEATURE = "surface_pattern"
MODEL_FEATURES = FEATURE_NAMES + [SURFACE_FEATURE]
DEFAULT_BANDS = {"low": 0.35, "high": 0.65}
MIN_WORDS_HARD = 10  # below this we refuse to score
MIN_WORDS_SOFT = 50  # below this we warn
MIN_BENGALI_HARD = 0.30  # below this share of Bengali letters we refuse to score
MIN_BENGALI_SOFT = 0.70  # below this we warn (mixed-language text)

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
)


# --------------------------------------------------------------------------- #
# Feature matrices
# --------------------------------------------------------------------------- #
def _docs(texts) -> list[Doc]:
    return [tokenize(t) for t in texts]


def build_train_features(docs: list[Doc], labels: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    """Cross-fitted features for TRAINING rows (perplexity + surface score are out-of-fold)."""
    n_splits = max(2, min(n_splits, len(set(groups))))
    rows = [base_features(d) for d in docs]
    ppl = np.zeros(len(docs))
    surf = np.zeros(len(docs))
    for fit_idx, score_idx in GroupKFold(n_splits=n_splits).split(docs, labels, groups):
        human_docs = [docs[i] for i in fit_idx if labels[i] == 0]
        lm = train_perplexity_model_from_docs(human_docs)
        for i in score_idx:
            ppl[i] = perplexity(docs[i], lm)
        sm = SurfaceModel().fit([docs[i].text for i in fit_idx], labels[fit_idx])
        surf[score_idx] = sm.score([docs[i].text for i in score_idx])
    for r, p, s in zip(rows, ppl, surf):
        r["perplexity"] = float(p)
        r[SURFACE_FEATURE] = float(s)
    return pd.DataFrame(rows)[MODEL_FEATURES]


def build_eval_features(docs: list[Doc], lm, surface: SurfaceModel) -> pd.DataFrame:
    rows = [features_from_doc(d, lm) for d in docs]
    scores = surface.score([d.text for d in docs])
    for r, s in zip(rows, scores):
        r[SURFACE_FEATURE] = float(s)
    return pd.DataFrame(rows)[MODEL_FEATURES]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def group_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, df["label"], groups=df["pair_id"]))
    return train_idx, test_idx


def exclude_pairs_by_source(df: pd.DataFrame, sources: list[str], log=print) -> pd.DataFrame:
    """Drop every pair whose HUMAN row came from one of `sources` (twin AI rows go too)."""
    if not sources:
        return df
    want = {s.lower() for s in sources}
    human_src = df[df["label"] == 0].set_index("pair_id")["source"].astype(str).str.lower()
    bad_pairs = set(human_src[human_src.isin(want)].index)
    out = df[~df["pair_id"].isin(bad_pairs)].reset_index(drop=True)
    log(f"excluded {len(df) - len(out)} rows from {len(bad_pairs)} pairs whose human text came from {sorted(want)}")
    return out


def _evaluate(model, X_test, y_test, test_df) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    report = classification_report(y_test, pred, target_names=["human", "ai"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    human_mask = (y_test == 0).to_numpy()
    fpr_at = {}
    for thr in (0.5, DEFAULT_BANDS["high"], 0.8):
        fpr_at[str(thr)] = float((proba[human_mask] >= thr).mean()) if human_mask.any() else None
    metrics = {
        "accuracy": float((pred == y_test.to_numpy()).mean()),
        "roc_auc": float(roc_auc_score(y_test, proba)) if len(set(y_test)) == 2 else None,
        "ai_precision": float(report["ai"]["precision"]),
        "ai_recall": float(report["ai"]["recall"]),
        "human_recall": float(report["human"]["recall"]),
        "confusion_matrix": {"rows": "true [human, ai]", "cols": "pred [human, ai]", "values": cm.tolist()},
        "human_false_positive_rate_at": fpr_at,
        "report": report,
    }
    tdf = test_df.reset_index(drop=True).copy()
    tdf["pred"] = pred
    metrics["accuracy_by_origin"] = {
        str(o): {"n": int(len(s)), "accuracy": float((s["pred"] == s["label"]).mean())}
        for o, s in tdf.groupby("origin")
    }
    # Per-source breakdown: does it catch every generator, and leave every kind of human text alone?
    tdf["src"] = tdf["source"].astype(str).str.lower()
    metrics["correct_rate_by_source"] = {
        f"{'ai' if lab == 1 else 'human'}:{src}": {"n": int(len(s)), "correct_rate": float((s["pred"] == lab).mean())}
        for (lab, src), s in tdf.groupby(["label", "src"])
    }
    return metrics


def _reference_stats(X: pd.DataFrame, y: pd.Series) -> dict:
    return {
        name: {"human_median": float(X.loc[y == 0, name].median()), "ai_median": float(X.loc[y == 1, name].median())}
        for name in MODEL_FEATURES
    }


@dataclass
class TrainResult:
    model: xgb.XGBClassifier
    lm: object
    surface: SurfaceModel
    meta: dict


def train(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, refit_all: bool = False,
          exclude_sources: list[str] | None = None, log=print) -> TrainResult:
    t0 = time.time()
    df = exclude_pairs_by_source(df.reset_index(drop=True), exclude_sources or [], log)
    train_idx, test_idx = group_split(df, test_size, seed)
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    log(f"split by pair_id: {len(train_df)} train rows / {len(test_df)} test rows")

    log("tokenizing ...")
    train_docs = _docs(train_df["text"])
    test_docs = _docs(test_df["text"])
    y_train, y_test = train_df["label"].reset_index(drop=True), test_df["label"].reset_index(drop=True)

    log("building cross-fitted training features (perplexity + surface score are out-of-fold) ...")
    X_train = build_train_features(train_docs, y_train.to_numpy(), train_df["pair_id"].to_numpy())
    final_lm = train_perplexity_model_from_docs([d for d, y in zip(train_docs, y_train) if y == 0])
    final_surface = SurfaceModel().fit([d.text for d in train_docs], y_train)
    X_test = build_eval_features(test_docs, final_lm, final_surface)

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)

    metrics = _evaluate(model, X_test, y_test, test_df.reset_index(drop=True))
    ref = _reference_stats(X_train, y_train)

    if refit_all:
        log("refitting on ALL data for the shipped model (the metrics above stay from the held-out split) ...")
        all_docs = train_docs + test_docs
        y_all = pd.concat([y_train, y_test], ignore_index=True)
        groups_all = np.concatenate([train_df["pair_id"].to_numpy(), test_df["pair_id"].to_numpy()])
        X_all = build_train_features(all_docs, y_all.to_numpy(), groups_all)
        final_lm = train_perplexity_model_from_docs([d for d, y in zip(all_docs, y_all) if y == 0])
        final_surface = SurfaceModel().fit([d.text for d in all_docs], y_all)
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_all, y_all)
        ref = _reference_stats(X_all, y_all)

    meta = {
        "version": MODEL_VERSION,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "feature_names": MODEL_FEATURES,
        "bands": DEFAULT_BANDS,
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "refit_on_all_data": bool(refit_all),
        "exclude_sources": sorted(s.lower() for s in (exclude_sources or [])),
        "label_balance": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "origins": {str(k): int(v) for k, v in df["origin"].value_counts().items()},
        "metrics": metrics,
        "reference": ref,
        "train_seconds": round(time.time() - t0, 1),
    }
    return TrainResult(model=model, lm=final_lm, surface=final_surface, meta=meta)


def save(result: TrainResult, model_dir: str | Path = "models") -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, model_dir / "model.joblib", compress=3)
    joblib.dump(result.lm, model_dir / "lm.joblib", compress=3)
    joblib.dump(result.surface, model_dir / "surface.joblib", compress=3)
    (model_dir / "meta.json").write_text(json.dumps(result.meta, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Inference + explanation (spec Section 5.5)
# --------------------------------------------------------------------------- #
FEATURE_INFO = {
    "burstiness": ("Sentence rhythm", "how much sentence length varies (std-dev, in words)"),
    "perplexity": ("Predictability", "how surprising the word sequence is versus human writing"),
    "ttr": ("Vocabulary variety", "unique words divided by total words"),
    "stopword_ratio": ("Function-word share", "fraction of words that are common function words"),
    "punct_ratio": ("Punctuation density", "punctuation marks as a share of all tokens"),
    "avg_word_len": ("Word length", "average characters per word"),
    "mean_sent": ("Average sentence length", "mean words per sentence"),
    "max_sent": ("Longest sentence", "words in the longest sentence"),
    "hapax": ("One-off words", "share of words that appear only once"),
    "yule_k": ("Word repetition", "Yule's K: how concentrated the vocabulary is on repeated words"),
    "entropy": ("Vocabulary spread", "Shannon entropy of the word distribution"),
    "long_word": ("Long-word share", "share of words with 8+ characters"),
    "comma": ("Comma use", "commas per 100 characters"),
    "danda": ("Sentence-ending marks", "danda (।) per 100 characters"),
    "quote": ("Quotation marks", "quote characters per 100 characters"),
    "digit": ("Digits", "digit characters per 100 characters"),
    "colon": ("Colons and semicolons", "per 100 characters"),
    "dash": ("Dashes and hyphens", "per 100 characters"),
    "paren": ("Brackets", "parentheses per 100 characters"),
    "conj": ("Connector words", "share of connectors such as এবং, কিন্তু, তাই"),
    SURFACE_FEATURE: ("Spelling and punctuation habits", "character-level patterns learned from human vs AI-rewritten samples"),
}


class ModelNotReady(RuntimeError):
    pass


class Predictor:
    def __init__(self, model_dir: str | Path = "models"):
        model_dir = Path(model_dir)
        needed = ["model.joblib", "lm.joblib", "surface.joblib", "meta.json"]
        missing = [f for f in needed if not (model_dir / f).exists()]
        if missing:
            raise ModelNotReady(f"Missing {missing} in {model_dir}/. Run scripts/train.py first.")
        self.model = joblib.load(model_dir / "model.joblib")
        self.lm = joblib.load(model_dir / "lm.joblib")
        self.surface = joblib.load(model_dir / "surface.joblib")
        self.meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        self._booster = self.model.get_booster()
        self.bands = self.meta.get("bands", DEFAULT_BANDS)

    # -- helpers ---------------------------------------------------------- #
    def band(self, p: float) -> dict:
        if p >= self.bands["high"]:
            return {"key": "review", "label": "Worth a closer look",
                    "detail": "Several measurable patterns resemble AI-written text. This is a prompt for a conversation, not proof."}
        if p <= self.bands["low"]:
            return {"key": "human_like", "label": "Reads more like human writing",
                    "detail": "The measured patterns sit closer to human-written samples. No detector is perfect."}
        return {"key": "inconclusive", "label": "Inconclusive",
                "detail": "The patterns are mixed. The score alone should not be used to judge this text."}

    def _reason(self, name: str, value: float, shap_value: float, total_abs: float, text: str) -> dict:
        label, meaning = FEATURE_INFO[name]
        ref = self.meta["reference"][name]
        direction = "ai" if shap_value > 0 else "human"
        toward = "AI-written" if direction == "ai" else "human-written"
        out = {
            "feature": name,
            "label": label,
            "meaning": meaning,
            "value": round(float(value), 3),
            "direction": direction,
            "impact": round(float(abs(shap_value) / total_abs * 100), 1) if total_abs else 0.0,
            "shap": round(float(shap_value), 3),
            "examples": [],
            "text": (
                f"{label} ({value:.2f}) points toward {toward} text. "
                f"In the training data, human samples centre near {ref['human_median']:.2f} "
                f"and AI samples near {ref['ai_median']:.2f}."
            ),
        }
        if name == SURFACE_FEATURE:
            ex = self.surface.explain(text, k=3)[direction]
            out["examples"] = ex
            out["text"] = (
                f"The character-level habits in this text lean {toward}. "
                + (f"Patterns that contributed most: {', '.join('«' + e + '»' for e in ex)}." if ex else "")
            )
        return out

    # -- main entry ------------------------------------------------------- #
    def analyze(self, text: str, top_k: int = 3) -> dict:
        doc = tokenize(text)
        stats = doc_stats(doc)
        if stats["words"] < MIN_WORDS_HARD:
            raise ValueError(f"Text too short: need at least {MIN_WORDS_HARD} words, got {stats['words']}.")
        ratio = bengali_ratio(text)
        if ratio < MIN_BENGALI_HARD:
            raise ValueError("This does not look like Bangla text. Prohori only scores Bangla.")

        row = features_from_doc(doc, self.lm)
        row[SURFACE_FEATURE] = float(self.surface.score([doc.text])[0])
        feats = pd.DataFrame([row])[MODEL_FEATURES]
        p = float(self.model.predict_proba(feats)[0, 1])
        # Exact TreeSHAP values computed natively by XGBoost (last column is the bias term).
        sv = self._booster.predict(xgb.DMatrix(feats), pred_contribs=True)[0][:-1]
        total_abs = float(np.abs(sv).sum())
        ranked = sorted(zip(MODEL_FEATURES, feats.iloc[0].tolist(), sv.tolist()), key=lambda x: abs(x[2]), reverse=True)
        reasons = [self._reason(n, v, s, total_abs, doc.text) for n, v, s in ranked[:top_k]]

        warnings = []
        if stats["words"] < MIN_WORDS_SOFT:
            warnings.append(f"Only {stats['words']} words: scores on very short text are unreliable. Prefer {MIN_WORDS_SOFT}+ words.")
        if stats["sentences"] < 3:
            warnings.append("Fewer than 3 sentences: the sentence-rhythm features are almost meaningless here.")
        if ratio < MIN_BENGALI_SOFT:
            warnings.append("Part of this text is not Bangla (mixed language). Scores on mixed-language text are less reliable.")

        return {
            "ai_probability": round(p, 3),
            "band": self.band(p),
            "top_reasons": [f"{r['feature']} ({'+' if r['direction'] == 'ai' else '-'}{abs(r['shap']):.2f})" for r in reasons],
            "reasons": reasons,
            "features": {k: round(float(v), 3) for k, v in feats.iloc[0].items()},
            "stats": stats,
            "warnings": warnings,
            "model": {"version": self.meta["version"], "trained_at": self.meta["trained_at"]},
            "disclaimer": "A probability, not a verdict. Treat a high score as the start of a conversation, never as proof.",
        }
