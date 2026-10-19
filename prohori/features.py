"""Feature engineering for Prohori (spec Section 5.3).

Six features, no LLM anywhere:
    burstiness, perplexity, ttr, stopword_ratio, punct_ratio, avg_word_len

Design notes
------------
* Tokenization is done ONCE per text (`tokenize`) and reused, because the
  perplexity feature is computed against several different language models
  during cross-fitting (see train.py).
* Perplexity uses a Laplace-smoothed trigram model (nltk.lm.Laplace). An
  unsmoothed model returns inf on unseen n-grams, which is useless as a feature.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from nltk.lm import Laplace
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.util import ngrams

from .bntok import PUNCTUATIONS, STOPWORDS as _STOPWORDS, NLTKTokenizer

# The spec's six features first, then extra explainable stylometric features that
# were measured to add signal on the real dataset (see CONNECT.md "Deviations").
FEATURE_NAMES = [
    "burstiness",
    "perplexity",
    "ttr",
    "stopword_ratio",
    "punct_ratio",
    "avg_word_len",
    "mean_sent",
    "max_sent",
    "hapax",
    "yule_k",
    "entropy",
    "long_word",
    "comma",
    "danda",
    "quote",
    "digit",
    "colon",
    "dash",
    "paren",
    "conj",
]

NGRAM_ORDER = 3
PERPLEXITY_CAP = 1000.0  # applied when the LM still returns inf / nan

_tokenizer = NLTKTokenizer()
STOPWORDS = set(_STOPWORDS)
PUNCT = set(PUNCTUATIONS)

_BENGALI_CHAR = re.compile(r"[\u0980-\u09FF]")


@dataclass
class Doc:
    """A tokenized text. Cheap to keep around and reuse."""

    words: list[str]  # word tokens, punctuation INCLUDED (as bnlp emits them)
    sent_lens: list[int]  # words per sentence (punctuation included, like the spec)
    text: str = ""  # the raw text (character-level features and the surface model need it)


def tokenize(text: str) -> Doc:
    text = (text or "").strip()
    if not text:
        return Doc(words=[], sent_lens=[], text="")
    sents = _tokenizer.sentence_tokenize(text)
    words = _tokenizer.word_tokenize(text)
    sent_lens = [len(_tokenizer.word_tokenize(s)) for s in sents]
    return Doc(words=words, sent_lens=sent_lens, text=text)


def bengali_ratio(text: str) -> float:
    """Share of letters in `text` that are Bengali script. Used for input warnings."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return len(_BENGALI_CHAR.findall(text)) / len(letters)


# --------------------------------------------------------------------------- #
# Language model (perplexity)
# --------------------------------------------------------------------------- #
def train_perplexity_model(human_texts, n: int = NGRAM_ORDER) -> Laplace:
    """Train a Laplace-smoothed n-gram LM on HUMAN text only."""
    return train_perplexity_model_from_docs([tokenize(t) for t in human_texts], n)


def train_perplexity_model_from_docs(docs: list[Doc], n: int = NGRAM_ORDER) -> Laplace:
    train_sents = [d.words for d in docs if d.words]
    if not train_sents:
        raise ValueError("Cannot train the perplexity model: no human text provided.")
    train_data, padded = padded_everygram_pipeline(n, train_sents)
    lm = Laplace(n)
    lm.fit(train_data, padded)
    return lm


def perplexity(doc: Doc, lm: Laplace, n: int = NGRAM_ORDER) -> float:
    if not doc.words:
        return PERPLEXITY_CAP
    padded = list(pad_both_ends(doc.words, n=n))
    value = lm.perplexity(list(ngrams(padded, n)))
    if not np.isfinite(value):
        return PERPLEXITY_CAP
    return float(min(value, PERPLEXITY_CAP))


# --------------------------------------------------------------------------- #
# Non-LM features
# --------------------------------------------------------------------------- #
def base_features(doc: Doc) -> dict[str, float]:
    """The five features that do not depend on a language model."""
    words = doc.words
    words_only = [w for w in words if w not in PUNCT]

    burstiness = float(np.std(doc.sent_lens)) if len(doc.sent_lens) > 1 else 0.0
    ttr = len(set(words_only)) / len(words_only) if words_only else 0.0
    punct_ratio = sum(1 for w in words if w in PUNCT) / len(words) if words else 0.0
    stopword_ratio = (
        sum(1 for w in words_only if w in STOPWORDS) / len(words_only) if words_only else 0.0
    )
    avg_word_len = float(np.mean([len(w) for w in words_only])) if words_only else 0.0

    feats = {
        "burstiness": burstiness,
        "ttr": ttr,
        "stopword_ratio": stopword_ratio,
        "punct_ratio": punct_ratio,
        "avg_word_len": avg_word_len,
    }
    feats.update(_extra_features(doc, words_only))
    return feats


_CONJ = {"এবং", "ও", "কিন্তু", "তবে", "তাই", "এছাড়াও", "তাছাড়া"}


def _extra_features(doc: Doc, words_only: list[str]) -> dict[str, float]:
    """Cheap, explainable stylometry. Deliberately excludes raw length features."""
    text = doc.text
    L = max(len(text), 1)
    n = len(words_only)
    sl = doc.sent_lens or [0]
    counts = Counter(words_only)
    per100 = lambda k: sum(text.count(c) for c in k) / L * 100  # noqa: E731
    return {
        "mean_sent": float(np.mean(sl)),
        "max_sent": float(max(sl)),
        "hapax": sum(1 for v in counts.values() if v == 1) / n if n else 0.0,
        "yule_k": 10000 * (sum(v * v for v in counts.values()) - n) / (n * n) if n > 1 else 0.0,
        "entropy": -sum(v / n * math.log(v / n) for v in counts.values()) if n else 0.0,
        "long_word": sum(1 for w in words_only if len(w) >= 8) / n if n else 0.0,
        "comma": per100(","),
        "danda": per100("\u0964"),
        "quote": per100("\"'\u2018\u2019\u201c\u201d"),
        "digit": sum(ch.isdigit() for ch in text) / L * 100,
        "colon": per100(":;"),
        "dash": per100("-\u2014\u2013"),
        "paren": per100("()"),
        "conj": sum(1 for w in words_only if w in _CONJ) / n if n else 0.0,
    }


def features_from_doc(doc: Doc, lm: Laplace) -> dict[str, float]:
    feats = base_features(doc)
    feats["perplexity"] = perplexity(doc, lm)
    return {name: feats[name] for name in FEATURE_NAMES}  # fixed column order


def extract_features(text: str, lm: Laplace, n: int = NGRAM_ORDER) -> dict[str, float]:
    """Spec-compatible entry point: text in, dict of 6 numbers out."""
    return features_from_doc(tokenize(text), lm)


def doc_stats(doc: Doc) -> dict[str, int]:
    words_only = [w for w in doc.words if w not in PUNCT]
    return {"words": len(words_only), "sentences": len(doc.sent_lens)}
