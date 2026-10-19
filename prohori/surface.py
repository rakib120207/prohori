"""Surface-pattern model: character n-gram TF-IDF + logistic regression.

Still classical statistics, still no LLM. Its output (one number) enters the
XGBoost model as the feature `surface_pattern`, so SHAP can explain it, and
`explain()` names the actual character patterns that drove the score.

It exists because the six stylometric features alone reached only ~62% on the
real dataset (see CONNECT.md). It captures spelling, punctuation and phrase
habits. The flip side: it can also latch onto dataset artifacts (e.g. archaic
spellings in WikiSource). train.py's --exclude-sources flag is how you control that.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def _visible(s: str) -> str:
    return s.replace("\u200c", "‹ZWNJ›").replace("\u200d", "‹ZWJ›").replace("\n", "↵")


class SurfaceModel:
    def __init__(self, ngram_range=(2, 4), min_df=3, max_features=100_000, C=3.0):
        self.vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, min_df=min_df,
            sublinear_tf=True, max_features=max_features,
        )
        self.clf = LogisticRegression(C=C, max_iter=2000)
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
        """The character patterns in `text` that pushed hardest toward AI / human."""
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
