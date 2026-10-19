"""End-to-end checks on FAKE data: proves the plumbing works, says nothing about real accuracy."""
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prohori.data import normalize_label, normalize_table  # noqa: E402
from prohori.features import FEATURE_NAMES, extract_features, tokenize, train_perplexity_model  # noqa: E402

PARA = "আমি ভাত খাই। তুমি কোথায় যাচ্ছো? আজ আবহাওয়া খুব সুন্দর, তাই আমরা বেড়াতে যাব। কাল বৃষ্টি হতে পারে।"


def test_features_are_six_finite_numbers():
    lm = train_perplexity_model([PARA, PARA + " আমরা সবাই বাজারে যাব।"])
    feats = extract_features("আমরা সবাই বাজারে যাব। সম্পূর্ণ অচেনা শব্দ আছে এখানে।", lm)
    assert list(feats) == FEATURE_NAMES
    assert all(math.isfinite(v) for v in feats.values())  # unseen words must not give inf


def test_empty_text_does_not_crash():
    lm = train_perplexity_model([PARA])
    feats = extract_features("", lm)
    assert all(math.isfinite(v) for v in feats.values())


def test_label_normalization():
    assert normalize_label("Human") == 0 and normalize_label("AI") == 1
    assert normalize_label("Human-written") == 0 and normalize_label("AI-rewritten") == 1
    assert normalize_label(0) == 0 and normalize_label(1) == 1
    assert normalize_label("weird") is None
    assert normalize_label("class_b", ai_label="class_b") == 1


def test_wide_format_is_melted():
    df = pd.DataFrame({
        "Human_Text": [f"{PARA} নম্বর {i} এর কথা।" for i in range(3)],
        "AI_Rewritten_Text": [f"{PARA} আরও কিছু কথা এখানে আছে {i}।" for i in range(3)],
    })
    out = normalize_table(df, "seed", verbose=False)
    assert set(out["label"]) == {0, 1} and out["pair_id"].nunique() == 3


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    py = sys.executable
    run = lambda *a: subprocess.run([py, *a], cwd=ROOT, check=True, capture_output=True, text=True)
    run("scripts/make_demo_data.py", "--pairs", "150", "--out", str(work / "raw" / "demo.csv"))
    run("scripts/prepare_dataset.py", "--raw-dir", str(work / "raw"), "--own-dir", str(work / "none"),
        "--out", str(work / "dataset.csv"))
    run("scripts/train.py", "--data", str(work / "dataset.csv"), "--model-dir", str(work / "models"))
    return work


def test_training_produces_artifacts_and_honest_metrics(trained):
    import json

    meta = json.loads((trained / "models" / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_train"] + meta["n_test"] == meta["n_rows"]
    assert 0.5 <= meta["metrics"]["accuracy"] <= 1.0
    for name in ("model.joblib", "lm.joblib", "surface.joblib"):
        assert (trained / "models" / name).exists()


def test_split_keeps_pairs_together(trained):
    from prohori.model import group_split

    df = pd.read_csv(trained / "dataset.csv")
    tr, te = group_split(df)
    assert not set(df.iloc[tr]["pair_id"]) & set(df.iloc[te]["pair_id"])


def test_predictor_and_api(trained, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODEL_DIR", str(trained / "models"))
    import importlib
    import api.main as main

    importlib.reload(main)
    df = pd.read_csv(trained / "dataset.csv")
    with TestClient(main.app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        text = df[df["label"] == 0]["text"].iloc[0]
        r = client.post("/analyze", json={"text": text})
        assert r.status_code == 200
        j = r.json()
        assert 0 <= j["ai_probability"] <= 1
        assert len(j["top_reasons"]) == 3 and len(j["reasons"]) == 3
        assert j["band"]["key"] in {"review", "human_like", "inconclusive"}
        assert "verdict" not in j  # design principle: never a bare verdict
        assert client.post("/analyze", json={"text": "ছোট লেখা"}).status_code == 422
        english = "this is plainly english text that is definitely long enough to pass the minimum word count check"
        assert client.post("/analyze", json={"text": english}).status_code == 422  # not Bangla: refuse
        assert client.get("/model-info").status_code == 200
        assert client.get("/").status_code == 200  # frontend served


def test_english_file_in_raw_dir_is_skipped(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8")
    run("scripts/make_demo_data.py", "--pairs", "30", "--out", str(raw / "bn.csv"))
    bn = pd.read_csv(raw / "bn.csv")
    bn.to_excel(raw / "bangla.xlsx", index=False)  # xlsx like the real download
    (raw / "bn.csv").unlink()
    en = pd.DataFrame({
        "Pair_ID": range(20), "Text": ["This is a perfectly ordinary english sentence about the weather today and more."] * 20,
        "Label": ["Human"] * 10 + ["AI"] * 10, "Source": "x",
    })
    en.loc[:, "Text"] = [f"{t} number {i}" for i, t in enumerate(en["Text"])]
    en.to_excel(raw / "english.xlsx", index=False)
    out = tmp_path / "ds.csv"
    res = run("scripts/prepare_dataset.py", "--raw-dir", str(raw), "--own-dir", str(tmp_path / "none"), "--out", str(out))
    assert "SKIPPING english.xlsx" in res.stdout
    ds = pd.read_csv(out)
    assert len(ds) == 60 and ds["text"].str.contains("english").sum() == 0
