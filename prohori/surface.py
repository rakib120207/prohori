"""Surface-pattern models: TF-IDF n-grams + logistic regression (classical, no LLM).

Two flavours, both fed into XGBoost as features so the explanations stay SHAP-based:
  * kind="char": character n-grams  -> spelling, punctuation and morphology habits
  * kind="word": word 1-2 grams     -> phrase and connector habits

`explain()` names the actual patterns in a text that pushed hardest toward AI / human.
Caveat: these models can also latch onto dataset artifacts (e.g. archaic spelling in
WikiSource); train.py's --exclude-sources flag is how that is controlled.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _visible(s: str) -> str:
    return s.replace("\u200c", "‹ZWNJ›").replace("\u200d", "‹ZWJ›").replace("\n", "↵")


def _make_vectorizer(kind: str) -> TfidfVectorizer:
    if kind == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True, max_features=100_000)
    if kind == "word":
        return TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2), min_df=2,
                               sublinear_tf=True, max_features=100_000)
    raise ValueError(f"unknown surface model kind {kind!r}")


class SurfaceModel:
    def __init__(self, kind: str = "char", C: float | None = None):
        self.kind = kind
        self.vec = _make_vectorizer(kind)
        self.clf = LogisticRegression(C=C if C is not None else (3.0 if kind == "char" else 10.0), max_iter=3000)
        self._names = None

    def fit(self, texts, y) -> "SurfaceModel":
        X = self.vec.fit_transform(list(texts))
        self.clf.fit(X, np.asarray(y))
        self._names = None
        return self

    def score(self, texts) -> np.ndarray:
        """Log-odds of AI. Positive = AI-like."""
        return self.clf.decision_function(self.vec.transform(list(texts)))

    def explain(self, text: str, k: int = 3) -> dict[str, list[str]]:
        if self._names is None:
            self._names = np.asarray(self.vec.get_feature_names_out())
        x = self.vec.transform([text])
        contrib = x.multiply(self.clf.coef_[0]).tocoo()
        pairs = sorted(zip(contrib.col, contrib.data), key=lambda t: t[1])

        def pick(items):
            chosen: list[str] = []
            for col, _ in items:
                g = self._names[col].strip()
                if len(g) < 2 or any(g in c or c in g for c in chosen):
                    continue  # skip whitespace-only and overlapping n-grams
                chosen.append(g)
                if len(chosen) == k:
                    break
            return [_visible(g) for g in chosen]

        return {"ai": pick(reversed(pairs)), "human": pick(pairs)}


SURFACE_KINDS = ("char", "word")


def fit_surfaces(texts, y) -> dict[str, SurfaceModel]:
    texts = list(texts)
    return {k: SurfaceModel(k).fit(texts, y) for k in SURFACE_KINDS}
