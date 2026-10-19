"""Lightweight Bangla tokenizer + corpora, a drop-in for the parts of bnlp that Prohori uses.

Why this exists: `import bnlp` pulls in gensim, sentencepiece, fasttext-style
dependencies and hundreds of MB of RAM/disk, none of which scoring needs. The logic
below is the same as bnlp's NLTKTokenizer (swap danda for '.', protect real dots,
use NLTK's punkt), and tests/test_tokenizer.py checks token-for-token equality with
bnlp whenever bnlp is installed.
"""
from __future__ import annotations

import os
from pathlib import Path

import nltk

_DUMMY = "XTEMPDOT"
PUNCTUATIONS = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~।ঃ"
STOPWORDS = frozenset(
    (Path(__file__).parent / "resources" / "bn_stopwords.txt").read_text(encoding="utf-8").split("\n")
)


def _ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError as e:  # no network and no baked-in data
            raise RuntimeError(
                "NLTK 'punkt_tab' data is missing. Run: python -c \"import nltk; nltk.download('punkt_tab')\""
            ) from e


_ensure_punkt()


def _protect(text: str) -> str:
    return text.replace(".", _DUMMY).replace("।", ".")


def _restore(token: str) -> str:
    return token.replace(".", "।").replace(_DUMMY, ".")


class NLTKTokenizer:
    def word_tokenize(self, text: str) -> list[str]:
        return [_restore(t) for t in nltk.word_tokenize(_protect(text))]

    def sentence_tokenize(self, text: str) -> list[str]:
        return [_restore(t) for t in nltk.tokenize.sent_tokenize(_protect(text))]
