"""Prohori API (spec Section 5.6).

  POST /analyze   {"text": "..."} -> probability + ranked, plain-language reasons
  GET  /health    liveness + whether a model is loaded
  GET  /model-info  training metadata and held-out metrics

The live scoring path never calls an LLM. It runs: tokenize -> 6 features -> XGBoost -> SHAP.
The same app also serves frontend/index.html at "/" so one process is a complete product.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from prohori.model import ModelNotReady, Predictor

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))
FRONTEND_DIR = ROOT / "frontend"
MAX_CHARS = int(os.environ.get("MAX_CHARS", "20000"))

state: dict = {"predictor": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["predictor"] = Predictor(MODEL_DIR)
    except ModelNotReady as e:
        state["error"] = str(e)
    yield


app = FastAPI(title="Prohori (প্রহরী) - Bangla AI-text detector", version="1.0.0", lifespan=lifespan)

origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class TextIn(BaseModel):
    text: str = Field(..., description="Bangla text to analyze")


def _predictor() -> Predictor:
    if state["predictor"] is None:
        raise HTTPException(503, state["error"] or "Model not loaded.")
    return state["predictor"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state["predictor"] is not None, "detail": state["error"]}


@app.get("/model-info")
def model_info():
    pred = _predictor()
    meta = pred.meta
    out = {k: meta[k] for k in ("version", "trained_at", "n_rows", "n_train", "n_test", "label_balance", "origins", "metrics", "bands")}
    out["cv_report"] = pred.cv_report  # robust cross-validated numbers, if models/cv_report.json exists
    return out


@app.post("/analyze")
def analyze(payload: TextIn):
    text = payload.text.strip()
    if len(text) > MAX_CHARS:
        raise HTTPException(413, f"Text too long ({len(text)} chars). Limit is {MAX_CHARS}.")
    try:
        return _predictor().analyze(text)
    except ValueError as e:  # too short
        raise HTTPException(422, str(e))


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")

    @app.get("/config.js", include_in_schema=False)
    def config_js():
        return FileResponse(FRONTEND_DIR / "config.js", media_type="application/javascript")
