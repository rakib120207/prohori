"""Training, scoring and honest evaluation for Prohori (spec Sections 5.4 and 5.5).

Deliberate deviations from the spec's sample code (all explained in CONNECT.md):

1. Split by Pair_ID, not by row (human/AI twins never straddle train and test).
2. Every learned feature is cross-fitted: perplexity and both surface scores are
   out-of-fold for training rows.
3. Extra explainable stylometric features + two surface-pattern scores (char and word
   n-grams), because the spec's six features alone reached only ~62% on the real data.
4. --exclude-sources, to drop a confounded sub-corpus (WikiSource's archaic spelling).
5. CANONICALIZATION (canon.py): spelling variants, typography and formal/colloquial synonym
   pairs are collapsed before any feature, so "polished" writing is not mistaken for AI.
6. CHUNKED SCORING. Training texts average ~60 words. Scoring a 500-word essay as one
   block drifts out of that distribution (measured: human false flags rose from ~8% to
   ~23%). Long texts are therefore split into ~60-word chunks, each chunk is scored, and
   the chunk log-odds are averaged. The per-chunk scores double as a paragraph view.
7. Evidence levels: how much text was scored decides how strongly the app may speak.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from .bntok import NLTKTokenizer
from .canon import canonicalize
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
from .surface import SURFACE_KINDS, SurfaceModel, fit_surfaces

MODEL_VERSION = "prohori-v2.1"
SURFACE_FEATURES = {"char": "surface_pattern", "word": "surface_words"}
MODEL_FEATURES = FEATURE_NAMES + [SURFACE_FEATURES[k] for k in SURFACE_KINDS]
# Measured in cross-validation: when confidence >= 0.8 (p >= 0.8 or p <= 0.2) the model is right ~91% of the time
# and answers on ~62% of ~60-word texts. Precision over coverage, because a false accusation is the costly error.
DEFAULT_BANDS = {"low": 0.2, "high": 0.8}

MIN_WORDS_HARD = 10  # below this we refuse to score
MIN_BENGALI_HARD = 0.30  # below this share of Bengali letters we refuse to score
MIN_BENGALI_SOFT = 0.70  # below this we warn (mixed-language text)

CHUNK_TARGET_WORDS = 60  # matches the average training text length
CHUNK_MIN_TOTAL = 100  # below this the text is scored as a single block
CHUNK_MIN_TAIL = 25

# (min words, key, label, detail)
EVIDENCE_LEVELS = [
    (250, "high", "Strong evidence base", "Plenty of text: the score averages many independent passages."),
    (100, "moderate", "Moderate evidence base", "Enough text for a useful reading; more text makes it steadier."),
    (50, "low", "Limited evidence", "Short text: expect roughly 1 in 5 mistakes."),
    (0, "very_low", "Too short to judge", "Under 50 words accuracy is only about 3 in 4 at best, so nothing is flagged."),
]

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.5,
    min_child_weight=3,
    reg_lambda=3,
    eval_metric="logloss",
    random_state=42,
)

_sent_tok = NLTKTokenizer()


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def split_chunks(text: str, target: int = CHUNK_TARGET_WORDS, min_total: int = CHUNK_MIN_TOTAL,
                 min_tail: int = CHUNK_MIN_TAIL) -> list[str]:
    """Group sentences into ~`target`-word chunks. Texts under `min_total` words stay whole."""
    text = text.strip()
    if len(text.split()) < min_total:
        return [text]
    pieces: list[str] = []
    for s in _sent_tok.sentence_tokenize(text):
        w = s.split()
        if len(w) > 2 * target:  # one huge unpunctuated "sentence": cut it by words
            pieces += [" ".join(w[i:i + target]) for i in range(0, len(w), target)]
        elif w:
            pieces.append(s)
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for s in pieces:
        cur.append(s)
        n += len(s.split())
        if n >= target:
            chunks.append(" ".join(cur))
            cur, n = [], 0
    if cur:
        if chunks and n < min_tail:
            chunks[-1] += " " + " ".join(cur)
        else:
            chunks.append(" ".join(cur))
    return chunks or [text]


# --------------------------------------------------------------------------- #
# Feature matrices
# --------------------------------------------------------------------------- #
def _docs(texts) -> list[Doc]:
    return [tokenize(t) for t in texts]


def build_train_features(docs: list[Doc], labels: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> pd.DataFrame:
    """Cross-fitted features for TRAINING rows (perplexity + surface scores are out-of-fold)."""
    n_splits = max(2, min(n_splits, len(set(groups))))
    rows = [base_features(d) for d in docs]
    ppl = np.zeros(len(docs))
    surf = {k: np.zeros(len(docs)) for k in SURFACE_KINDS}
    for fit_idx, score_idx in GroupKFold(n_splits=n_splits).split(docs, labels, groups):
        lm = train_perplexity_model_from_docs([docs[i] for i in fit_idx if labels[i] == 0])
        for i in score_idx:
            ppl[i] = perplexity(docs[i], lm)
        fit_texts = [docs[i].text for i in fit_idx]
        score_texts = [docs[i].text for i in score_idx]
        for kind in SURFACE_KINDS:
            surf[kind][score_idx] = SurfaceModel(kind).fit(fit_texts, labels[fit_idx]).score(score_texts)
    for j, r in enumerate(rows):
        r["perplexity"] = float(ppl[j])
        for kind in SURFACE_KINDS:
            r[SURFACE_FEATURES[kind]] = float(surf[kind][j])
    return pd.DataFrame(rows)[MODEL_FEATURES]


def build_eval_features(docs: list[Doc], lm, surfaces: dict[str, SurfaceModel]) -> pd.DataFrame:
    rows = [features_from_doc(d, lm) for d in docs]
    texts = [d.text for d in docs]
    for kind in SURFACE_KINDS:
        scores = surfaces[kind].score(texts)
        for r, s in zip(rows, scores):
            r[SURFACE_FEATURES[kind]] = float(s)
    return pd.DataFrame(rows)[MODEL_FEATURES]


def fit_pipeline(docs: list[Doc], y: np.ndarray, groups: np.ndarray):
    """Fit everything on a set of documents. Returns (model, lm, surfaces, X_train)."""
    X = build_train_features(docs, y, groups)
    lm = train_perplexity_model_from_docs([d for d, lab in zip(docs, y) if lab == 0])
    surfaces = fit_surfaces([d.text for d in docs], y)
    model = xgb.XGBClassifier(**XGB_PARAMS).fit(X, y)
    return model, lm, surfaces, X


# --------------------------------------------------------------------------- #
# Scoring one text (chunked)
# --------------------------------------------------------------------------- #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class TextScore:
    probability: float
    chunks: list[dict]
    mean_shap: np.ndarray  # mean per-feature contribution (log-odds) across chunks
    mean_features: pd.Series
    spread: float


class Scorer:
    def __init__(self, model, lm, surfaces):
        self.model, self.lm, self.surfaces = model, lm, surfaces
        self._booster = model.get_booster()

    def score_text(self, text: str, chunked: bool = True) -> TextScore:
        chunks = split_chunks(text) if chunked else [text.strip()]
        docs = _docs(chunks)
        X = build_eval_features(docs, self.lm, self.surfaces)
        contribs = self._booster.predict(xgb.DMatrix(X), pred_contribs=True)  # exact TreeSHAP; last col = bias
        margins = contribs.sum(axis=1)
        p_chunks = _sigmoid(margins)
        p_doc = float(_sigmoid(margins.mean()))
        return TextScore(
            probability=p_doc,
            chunks=[{"text": c, "probability": round(float(p), 3), "words": len(c.split())} for c, p in zip(chunks, p_chunks)],
            mean_shap=contribs[:, :-1].mean(axis=0),
            mean_features=X.mean(axis=0),
            spread=float(np.std(p_chunks)) if len(p_chunks) > 1 else 0.0,
        )

    def proba_single(self, texts: list[str]) -> np.ndarray:
        """Probability for each text scored as ONE block (how the training rows are scored)."""
        X = build_eval_features(_docs(texts), self.lm, self.surfaces)
        return self.model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------- #
# Training / evaluation
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


LENGTH_BUCKETS = [(0, 40, "under 40 words"), (40, 60, "40-60 words"), (60, 90, "60-90 words"), (90, 10**6, "over 90 words")]


def _by_length(words: np.ndarray, y: np.ndarray, p: np.ndarray) -> dict:
    out = {}
    for lo, hi, name in LENGTH_BUCKETS:
        m = (words > lo) & (words <= hi)
        if m.sum() >= 10:
            hum = m & (y == 0)
            out[name] = {
                "n": int(m.sum()),
                "accuracy": round(float(((p[m] >= 0.5) == y[m]).mean()), 3),
                "human_false_flag_rate": round(float((p[hum] >= DEFAULT_BANDS["high"]).mean()), 3) if hum.any() else None,
            }
    return out


def _selective(y: np.ndarray, p: np.ndarray) -> list[dict]:
    """If the app only answers when it is at least this confident: how much does it cover, and how right is it?"""
    conf = np.maximum(p, 1 - p)
    rows = []
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        m = conf >= t
        if m.sum() >= 10:
            rows.append({"min_confidence": t, "coverage": round(float(m.mean()), 3),
                         "accuracy_when_answering": round(float(((p[m] >= 0.5) == y[m]).mean()), 3)})
    return rows


def _evaluate(y_test: pd.Series, proba: np.ndarray, test_df: pd.DataFrame) -> dict:
    pred = (proba >= 0.5).astype(int)
    report = classification_report(y_test, pred, target_names=["human", "ai"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    y = y_test.to_numpy()
    human_mask = y == 0
    fpr_at = {str(t): (float((proba[human_mask] >= t).mean()) if human_mask.any() else None)
              for t in (0.5, DEFAULT_BANDS["high"], 0.8)}
    tdf = test_df.reset_index(drop=True).copy()
    tdf["pred"] = pred
    tdf["src"] = tdf["source"].astype(str).str.lower()
    return {
        "accuracy": float((pred == y).mean()),
        "roc_auc": float(roc_auc_score(y, proba)) if len(set(y)) == 2 else None,
        "ai_precision": float(report["ai"]["precision"]),
        "ai_recall": float(report["ai"]["recall"]),
        "human_recall": float(report["human"]["recall"]),
        "confusion_matrix": {"rows": "true [human, ai]", "cols": "pred [human, ai]", "values": cm.tolist()},
        "human_false_positive_rate_at": fpr_at,
        "report": report,
        "accuracy_by_origin": {str(o): {"n": int(len(s)), "accuracy": float((s["pred"] == s["label"]).mean())}
                               for o, s in tdf.groupby("origin")},
        "correct_rate_by_source": {
            f"{'ai' if lab == 1 else 'human'}:{src}": {"n": int(len(s)), "correct_rate": float((s["pred"] == lab).mean())}
            for (lab, src), s in tdf.groupby(["label", "src"])
        },
        "by_length": _by_length(tdf["text"].str.split().str.len().to_numpy(), y, proba),
        "selective": _selective(y, proba),
    }


def _reference_stats(X: pd.DataFrame, y: pd.Series) -> dict:
    return {name: {"human_median": float(X.loc[y == 0, name].median()), "ai_median": float(X.loc[y == 1, name].median())}
            for name in MODEL_FEATURES}


@dataclass
class TrainResult:
    model: xgb.XGBClassifier
    lm: object
    surface: dict
    meta: dict


def train(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42, refit_all: bool = False,
          exclude_sources: list[str] | None = None, log=print) -> TrainResult:
    t0 = time.time()
    df = exclude_pairs_by_source(df.reset_index(drop=True), exclude_sources or [], log)
    train_idx, test_idx = group_split(df, test_size, seed)
    train_df, test_df = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    log(f"split by pair_id: {len(train_df)} train rows / {len(test_df)} test rows")

    log("tokenizing ...")
    train_docs, test_docs = _docs(train_df["text"]), _docs(test_df["text"])
    log("fitting (cross-fitted features; this is the slow part) ...")
    model, lm, surfaces, X_train = fit_pipeline(train_docs, train_df["label"].to_numpy(), train_df["pair_id"].to_numpy())
    X_test = build_eval_features(test_docs, lm, surfaces)
    metrics = _evaluate(test_df["label"], model.predict_proba(X_test)[:, 1], test_df)
    ref = _reference_stats(X_train, train_df["label"])

    if refit_all:
        log("refitting on ALL data for the shipped model (the metrics above stay from the held-out split) ...")
        all_docs = train_docs + test_docs
        y_all = np.concatenate([train_df["label"].to_numpy(), test_df["label"].to_numpy()])
        g_all = np.concatenate([train_df["pair_id"].to_numpy(), test_df["pair_id"].to_numpy()])
        model, lm, surfaces, X_all = fit_pipeline(all_docs, y_all, g_all)
        ref = _reference_stats(X_all, pd.Series(y_all))

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
    return TrainResult(model=model, lm=lm, surface=surfaces, meta=meta)


def save(result: TrainResult, model_dir: str | Path = "models") -> None:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, model_dir / "model.joblib", compress=3)
    joblib.dump(result.lm, model_dir / "lm.joblib", compress=3)
    joblib.dump(result.surface, model_dir / "surface.joblib", compress=3)
    (model_dir / "meta.json").write_text(json.dumps(result.meta, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Honest evaluation: grouped cross-validation, unseen-generator test, long-text simulation
# --------------------------------------------------------------------------- #
def _long_doc_simulation(scorer: Scorer, test_df: pd.DataFrame, rng: random.Random, n_docs: int = 40) -> dict:
    """Concatenate k same-label test texts into long documents; compare whole-block vs chunked scoring.
    SYNTHETIC: the pieces are unrelated texts, which is easier than a real coherent essay."""
    out: dict = {}
    for k in (3, 6, 10):
        rows = []
        for lab in (0, 1):
            pool = test_df.loc[test_df["label"] == lab, "text"].tolist()
            if len(pool) < k:
                continue
            for _ in range(n_docs):
                txt = " ".join(rng.sample(pool, k))
                whole = float(scorer.proba_single([txt])[0])
                chunked = scorer.score_text(txt).probability
                rows.append((lab, len(txt.split()), whole, chunked))
        if not rows:
            continue
        r = pd.DataFrame(rows, columns=["y", "words", "whole", "chunk"])
        hum = r["y"] == 0
        out[str(k)] = {
            "avg_words": int(r["words"].mean()),
            "whole_block": {"accuracy": round(float(((r.whole >= .5) == r.y).mean()), 3),
                            "human_false_flag_rate": round(float((r.whole[hum] >= DEFAULT_BANDS["high"]).mean()), 3)},
            "chunked": {"accuracy": round(float(((r.chunk >= .5) == r.y).mean()), 3),
                        "human_false_flag_rate": round(float((r.chunk[hum] >= DEFAULT_BANDS["high"]).mean()), 3)},
        }
    return out


def cross_validate(df: pd.DataFrame, n_splits: int = 5, exclude_sources: list[str] | None = None,
                   seed: int = 7, log=print) -> dict:
    """5-fold grouped CV + leave-one-generator-out + long-text simulation. Takes a few minutes."""
    df = exclude_pairs_by_source(df.reset_index(drop=True), exclude_sources or [], log)
    y = df["label"].to_numpy()
    groups = df["pair_id"].to_numpy()
    docs = _docs(df["text"])
    P = np.zeros(len(df))
    rng = random.Random(seed)
    long_sim: list[dict] = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(df, y, groups)):
        log(f"  CV fold {k + 1}/{n_splits}")
        model, lm, surfaces, _ = fit_pipeline([docs[i] for i in tr], y[tr], groups[tr])
        scorer = Scorer(model, lm, surfaces)
        P[te] = model.predict_proba(build_eval_features([docs[i] for i in te], lm, surfaces))[:, 1]
        long_sim.append(_long_doc_simulation(scorer, df.iloc[te], rng))
    report: dict = {
        "n_rows": int(len(df)),
        "n_pairs": int(df["pair_id"].nunique()),
        "folds": n_splits,
        "pooled": _evaluate(df["label"], P, df),
    }
    merged: dict = {}
    for sim in long_sim:
        for kk, v in sim.items():
            merged.setdefault(kk, []).append(v)
    report["long_text_simulation"] = {
        "note": "SYNTHETIC: k unrelated same-label texts concatenated. Easier than a real essay; treat as an optimistic bound.",
        **{kk: {"avg_words": int(np.mean([v["avg_words"] for v in vs])),
                "whole_block": {m: round(float(np.mean([v["whole_block"][m] for v in vs])), 3) for m in ("accuracy", "human_false_flag_rate")},
                "chunked": {m: round(float(np.mean([v["chunked"][m] for v in vs])), 3) for m in ("accuracy", "human_false_flag_rate")}}
           for kk, vs in merged.items()},
    }

    # Leave-one-generator-out: how well does it catch an LLM it has never seen?
    ai_rows = df[df["label"] == 1].assign(src=lambda d: d["source"].astype(str).str.lower())
    logo = {}
    for gen, rows in ai_rows.groupby("src"):
        if len(rows) < 30:
            continue
        log(f"  leave-one-generator-out: {gen}")
        held = df["pair_id"].isin(set(rows["pair_id"])).to_numpy()
        tr, te = np.where(~held)[0], np.where(held)[0]
        model, lm, surfaces, _ = fit_pipeline([docs[i] for i in tr], y[tr], groups[tr])
        p = model.predict_proba(build_eval_features([docs[i] for i in te], lm, surfaces))[:, 1]
        yt = y[te]
        seen = P[te][yt == 1]  # same rows, scored by the cross-validated model that HAD seen this generator
        logo[gen] = {
            "n_ai": int((yt == 1).sum()),
            "ai_caught_when_unseen": round(float((p[yt == 1] >= 0.5).mean()), 3),
            "ai_caught_when_seen_in_training": round(float((seen >= 0.5).mean()), 3),
            "human_correct_on_twins": round(float((p[yt == 0] < 0.5).mean()), 3),
        }
    report["unseen_generator"] = logo

    # Leave-one-domain-out: how does it do on a kind of human text it has never seen?
    human_src = df[df["label"] == 0].set_index("pair_id")["source"].astype(str).str.lower()
    dom = df["pair_id"].map(human_src).fillna("")
    lodo = {}
    for d_name in sorted(set(dom) - {""}):
        held = (dom == d_name).to_numpy()
        if held.sum() < 60 or (~held).sum() < 200:
            continue
        log(f"  leave-one-domain-out: {d_name}")
        tr, te = np.where(~held)[0], np.where(held)[0]
        model, lm, surfaces, _ = fit_pipeline([docs[i] for i in tr], y[tr], groups[tr])
        p = model.predict_proba(build_eval_features([docs[i] for i in te], lm, surfaces))[:, 1]
        yt = y[te]
        lodo[d_name] = {
            "n": int(len(te)),
            "accuracy_when_unseen": round(float(((p >= 0.5) == yt).mean()), 3),
            "accuracy_when_seen_in_training": round(float(((P[te] >= 0.5) == yt).mean()), 3),
            "human_correct_when_unseen": round(float((p[yt == 0] < 0.5).mean()), 3),
            "human_flagged_at_0.8_when_unseen": round(float((p[yt == 0] >= DEFAULT_BANDS["high"]).mean()), 3),
        }
    report["unseen_domain"] = lodo
    return report


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
    "surface_pattern": ("Spelling and punctuation habits", "character-level patterns learned from human vs AI-rewritten samples"),
    "surface_words": ("Word and phrase habits", "word and word-pair patterns learned from human vs AI-rewritten samples"),
}


class ModelNotReady(RuntimeError):
    pass


def evidence_for(words: int) -> dict:
    for min_words, key, label, detail in EVIDENCE_LEVELS:
        if words >= min_words:
            return {"key": key, "label": label, "detail": detail, "words": words}
    raise AssertionError  # unreachable


class Predictor:
    def __init__(self, model_dir: str | Path = "models"):
        model_dir = Path(model_dir)
        needed = ["model.joblib", "lm.joblib", "surface.joblib", "meta.json"]
        missing = [f for f in needed if not (model_dir / f).exists()]
        if missing:
            raise ModelNotReady(f"Missing {missing} in {model_dir}/. Run scripts/train.py first.")
        self.scorer = Scorer(joblib.load(model_dir / "model.joblib"), joblib.load(model_dir / "lm.joblib"),
                             joblib.load(model_dir / "surface.joblib"))
        self.meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        cv_path = model_dir / "cv_report.json"
        self.cv_report = json.loads(cv_path.read_text(encoding="utf-8")) if cv_path.exists() else None
        self.bands = self.meta.get("bands", DEFAULT_BANDS)

    # -- helpers ---------------------------------------------------------- #
    def band(self, p: float, evidence_key: str) -> dict:
        if evidence_key == "very_low":
            return {"key": "inconclusive", "label": "Too short to judge",
                    "detail": "There is not enough text for a reliable reading. Paste at least 100 words."}
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
            "feature": name, "label": label, "meaning": meaning,
            "value": round(float(value), 3), "direction": direction,
            "impact": round(float(abs(shap_value) / total_abs * 100), 1) if total_abs else 0.0,
            "shap": round(float(shap_value), 3), "examples": [],
            "text": (f"{label} ({value:.2f}) points toward {toward} text. "
                     f"In the training data, human samples centre near {ref['human_median']:.2f} "
                     f"and AI samples near {ref['ai_median']:.2f}."),
        }
        for kind, fname in SURFACE_FEATURES.items():
            if name == fname:
                ex = self.scorer.surfaces[kind].explain(canonicalize(text), k=3)[direction]
                out["examples"] = ex
                what = "character-level habits" if kind == "char" else "word and phrase habits"
                out["text"] = (f"The {what} in this text lean {toward}. "
                               + (f"Patterns that contributed most: {', '.join('«' + e + '»' for e in ex)}." if ex else ""))
        return out

    # -- main entry ------------------------------------------------------- #
    def analyze(self, text: str, top_k: int = 3) -> dict:
        text = text.strip()
        stats = doc_stats(tokenize(text))
        if stats["words"] < MIN_WORDS_HARD:
            raise ValueError(f"Text too short: need at least {MIN_WORDS_HARD} words, got {stats['words']}.")
        ratio = bengali_ratio(text)
        if ratio < MIN_BENGALI_HARD:
            raise ValueError("This does not look like Bangla text. Prohori only scores Bangla.")

        sc = self.scorer.score_text(text)
        total_abs = float(np.abs(sc.mean_shap).sum())
        ranked = sorted(zip(MODEL_FEATURES, sc.mean_features.tolist(), sc.mean_shap.tolist()), key=lambda x: abs(x[2]), reverse=True)
        reasons = [self._reason(n, v, s, total_abs, text) for n, v, s in ranked[:top_k]]
        evidence = evidence_for(stats["words"])
        evidence["chunks"] = len(sc.chunks)

        warnings = []
        if stats["words"] < 100:
            warnings.append(f"Only {stats['words']} words. Measured accuracy is lowest on short text; 100+ words is much steadier.")
        if stats["sentences"] < 3:
            warnings.append("Fewer than 3 sentences: the sentence-rhythm features are almost meaningless here.")
        if ratio < MIN_BENGALI_SOFT:
            warnings.append("Part of this text is not Bangla (mixed language). Scores on mixed-language text are less reliable.")

        return {
            "ai_probability": round(sc.probability, 3),
            "band": self.band(sc.probability, evidence["key"]),
            "evidence": evidence,
            "top_reasons": [f"{r['feature']} ({'+' if r['direction'] == 'ai' else '-'}{abs(r['shap']):.2f})" for r in reasons],
            "reasons": reasons,
            "chunks": sc.chunks if len(sc.chunks) > 1 else [],
            "chunk_note": ("Each passage is scored on its own (about 60 words each), so single passages are less reliable than the overall score."
                           if len(sc.chunks) > 1 else None),
            "features": {k: round(float(v), 3) for k, v in sc.mean_features.items()},
            "stats": stats,
            "warnings": warnings,
            "model": {"version": self.meta["version"], "trained_at": self.meta["trained_at"]},
            "disclaimer": "A probability, not a verdict. Treat a high score as the start of a conversation, never as proof.",
        }
