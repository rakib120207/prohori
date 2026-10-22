"""Canonicalization: erase cosmetic differences BEFORE any feature is computed.

Why this exists (measured, see PUSHING_ACCURACY.md): the first models treated polish as
a sign of AI. Spelling conventions, typography and formal-vs-colloquial synonym choices
differed between the human samples (older textbooks and news) and the AI rewrites, so the
models learned "standard spelling + curly quotes + em dashes = AI". Merely polishing a
human text's spelling and punctuation raised its false-flag rate from 7.6% to 18.5%.

Everything below is applied identically at training time and at scoring time, so a text
scores the same whichever variant spelling or typography its author happened to use.
What is NOT touched: sentence structure, punctuation *habits* (commas, semicolons, sentence
length), vocabulary beyond the meaning-identical pairs listed here.
"""
from __future__ import annotations

import re
import unicodedata

# variant -> canonical. Only pairs with the SAME meaning (spelling variants, or a formal and a
# colloquial word for the same thing). Both directions collapse to one form.
VARIANTS = {
    # spelling conventions
    "উপর": "ওপর", "উপরে": "ওপরে", "উপরের": "ওপরের",
    "হিসাবে": "হিসেবে", "হিসাব": "হিসেব",
    "কোন": "কোনো", "ভাল": "ভালো", "কি": "কী", "হল": "হলো",
    "আরো": "আরও", "এখনো": "এখনও", "জন্যে": "জন্য",
    # formal vs colloquial words with identical meaning
    "সাথে": "সঙ্গে", "এই": "এ", "সেই": "সে", "এটি": "এটা", "ওটি": "ওটা",
    "একটি": "একটা", "এবং": "ও", "সকল": "সব",
}

# suffix alternations: বইটি/বইটা, বইগুলি/বইগুলো
_SUFFIXES = ((re.compile(r"টি$"), "টা"), (re.compile(r"গুলি$"), "গুলো"))
_BENGALI_RUN = re.compile(r"[\u0980-\u09FF]+")
_HONORIFIC = re.compile(r"(?<![\u0980-\u09FF])([তযএ]া|[তযএ])\u0981")  # তাঁর -> তার, যাঁর -> যার, এঁরা -> এরা
_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)
_PUNCT_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2014": "-", "\u2013": "-", "\u2015": "-", "\u2012": "-", "\u2212": "-",
    "\u2026": "...",
    "\u00a0": " ",
})
_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_SPACE_BEFORE_PUNCT = re.compile(r"[ \t]+(?=[।,;:?!)\]])")
_MULTISPACE = re.compile(r"\s+")


def _canon_word(m: re.Match) -> str:
    w = m.group(0)
    if w in VARIANTS:
        return VARIANTS[w]
    for pat, rep in _SUFFIXES:
        w = pat.sub(rep, w)
    return w


def canonicalize(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text)  # one Unicode form for য়, ড়, ঢ়
    t = t.translate(_ZERO_WIDTH).translate(_PUNCT_MAP).translate(_DIGITS)
    t = _HONORIFIC.sub(lambda m: m.group(1), t)
    t = _BENGALI_RUN.sub(_canon_word, t)
    t = _SPACE_BEFORE_PUNCT.sub("", t)
    return _MULTISPACE.sub(" ", t).strip()
