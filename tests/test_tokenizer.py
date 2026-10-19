"""prohori.bntok must tokenize exactly like bnlp did when the models were trained."""
import pytest

bnlp = pytest.importorskip("bnlp", reason="bnlp is optional; it is only used to prove equivalence")

from prohori.bntok import PUNCTUATIONS, STOPWORDS, NLTKTokenizer  # noqa: E402

TEXTS = [
    "আমি ভাত খাই। তুমি কোথায় যাচ্ছো? আজ আবহাওয়া খুব সুন্দর, তাই আমরা বেড়াতে যাব।",
    "ডা. রহমান ১.৫ কেজি চাল কিনলেন। দাম ছিল ২০০ টাকা!",
    "\"সে বলল, ‘আমি যাব’ — কিন্তু গেল না।\" (এটা সত্যি)",
]


def test_same_tokens_as_bnlp():
    from bnlp import NLTKTokenizer as Ref
    from bnlp.corpus.corpus import BengaliCorpus

    ref, mine = Ref(), NLTKTokenizer()
    for t in TEXTS:
        assert ref.word_tokenize(t) == mine.word_tokenize(t)
        assert ref.sentence_tokenize(t) == mine.sentence_tokenize(t)
    assert STOPWORDS == set(BengaliCorpus.stopwords)
    assert set(PUNCTUATIONS) == set(BengaliCorpus.punctuations)
