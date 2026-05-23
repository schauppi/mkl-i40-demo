"""
FastAPI Inference Server — Predictive Maintenance (Vent4)
=========================================================
This module serves a trained scikit-learn Random Forest model via HTTP.

Endpoints:
  GET  /health   → liveness probe (is the server up?)
  POST /predict  → batch inference (send feature rows, get back predictions)

Download the model (once):
  bash Project/Server/download_model.sh                                               (macOS/Linux)
  PowerShell -ExecutionPolicy Bypass -File Project\Server\download_model.ps1         (Windows)

Start the server:
  uvicorn Project.Server.inference_server:app --reload

The model file (rf_vent4.pkl) must live in the same directory as this file.
It is loaded once at process startup (module level), so every incoming
request reuses the same in-memory object — no per-request disk I/O.
"""

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# MODEL_PATH can be overridden with an environment variable, which makes it
# easy to point at a different model file when running inside Docker or on
# another machine without touching source code.
# The default resolves relative to this file so the server works regardless
# of the working directory.
_SERVER_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.getenv("MODEL_PATH", str(_SERVER_DIR / "rf_vent4.pkl"))

# Loading at module level is intentional: the model is loaded exactly once
# when uvicorn imports this module, not on every HTTP request. A typical
# scikit-learn pipeline can be 50–500 MB in memory; reloading it per request
# would add hundreds of milliseconds of latency and hammer the disk.
model = joblib.load(MODEL_PATH)

# `feature_names_in_` is set by scikit-learn when the model was fitted on a
# DataFrame. We capture the column order here so we can guarantee that the
# DataFrame we build from incoming JSON always has columns in the exact same
# order the model expects — even if the client sends keys in a different order.
feature_cols: list[str] = list(model.feature_names_in_)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(title="Predictive Maintenance — Vent Inference Server")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    Input schema for the /predict endpoint.

    `features` is a list of samples, where each sample is a dict mapping
    feature name → value. Using a list allows the client to send an entire
    batch in a single HTTP request instead of one request per row.

    Pydantic automatically validates that every value is a float and raises
    a clear HTTP 422 error if the payload is malformed — no manual validation
    code needed.

    Example payload:
        {
          "features": [
            {"feat_A": 1.2, "feat_B": 0.5, ...},
            {"feat_A": 3.1, "feat_B": 0.9, ...}
          ]
        }
    """
    features: list[dict[str, float]]


class PredictResponse(BaseModel):
    """
    Output schema for a single predicted sample.

    Fields:
      prediction        — 0 = normal, 1 = pre-failure (integer class label)
      label             — human-readable string version of prediction
      probability_normal     — model's confidence that the sample is class 0
      probability_pre_failure — model's confidence that the sample is class 1

    The two probabilities always sum to 1.0 (they come from predict_proba).
    """
    prediction: int
    label: str
    probability_normal: float
    probability_pre_failure: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Liveness probe — returns immediately with basic server information.

    This endpoint is intentionally cheap: it does no model work. Its purpose
    is to let monitoring tools, load balancers, or CI scripts confirm that the
    server process is running and the model was loaded successfully (if we got
    here, the module-level `joblib.load` already succeeded).
    """
    return {"status": "ok", "model": MODEL_PATH, "n_features": len(feature_cols)}


@app.post("/predict", response_model=list[PredictResponse])
def predict(request: PredictRequest):
    """
    Batch prediction endpoint.

    Accepts a list of feature dicts, runs them through the model in one
    vectorised call, and returns one PredictResponse per input sample.

    Steps:
      1. Convert the list of dicts to a DataFrame and select/reorder columns.
      2. Call model.predict()       → class labels (0 or 1) for every row.
      3. Call model.predict_proba() → probability matrix, shape (n_samples, 2).
      4. Zip predictions and probability rows together and build the response.
    """
    # Build a DataFrame and immediately select only the training features in
    # the correct order. This guards against clients that send extra columns
    # (which would cause sklearn to raise) or columns in the wrong order
    # (which would silently produce wrong predictions).
    df = pd.DataFrame(request.features)[feature_cols]

    predictions = model.predict(df)

    # predict_proba returns an (n_samples, n_classes) array.
    # Column 0 → probability of class 0 (normal)
    # Column 1 → probability of class 1 (pre-failure)
    probas = model.predict_proba(df)

    # Build one PredictResponse per row by zipping the flat predictions array
    # with the rows of the probas matrix.
    return [
        PredictResponse(
            prediction=int(pred),
            label="pre-failure" if pred == 1 else "normal",
            probability_normal=float(proba[0]),
            probability_pre_failure=float(proba[1]),
        )
        for pred, proba in zip(predictions, probas)
    ]
