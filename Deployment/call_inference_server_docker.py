"""
Demo Client — Containerized Inference Server
============================================
This script calls the FastAPI inference server running inside Docker.

Prerequisites (run once, in order):
  1. Download the model:
       bash Project/Deployment/download_files.sh

  2. Build and start the container:
       docker compose -f Project/Deployment/docker-compose.yml up --build

  3. Run this client (in a second terminal):
       python Project/Deployment/call_inference_server_docker.py

The inference subset CSV is downloaded from GitHub on the first run and cached
in the current directory — subsequent runs load it locally (no network needed).

BASE_URL can be overridden via environment variable to point at a remote host:
  BASE_URL=http://my-server:8000 python Project/Deployment/call_inference_server_docker.py
"""

import os
from pathlib import Path

import httpx
import pandas as pd

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

SUBSET_GITHUB_URL = (
    "https://media.githubusercontent.com/media/schauppi/UAS_DAVIS_SS2026_Datasets"
    "/refs/heads/main/Datasets/MKL_I4_0/vent_4_failure_data_engineered_inference.csv"
)

SUBSET_LOCAL_PATH = Path("vent_4_failure_data_engineered_inference.csv")


def load_subset() -> pd.DataFrame:
    if not SUBSET_LOCAL_PATH.exists():
        print("Downloading subset from GitHub …")
        df = pd.read_csv(SUBSET_GITHUB_URL)
        df.to_csv(SUBSET_LOCAL_PATH, index=False)
        print(f"Saved to {SUBSET_LOCAL_PATH.resolve()}")
    else:
        print(f"Loading cached subset from {SUBSET_LOCAL_PATH.resolve()} …")
    return pd.read_csv(SUBSET_LOCAL_PATH)


def main() -> None:
    # --- 1. Health check -------------------------------------------------------
    resp = httpx.get(f"{BASE_URL}/health")
    print(f"[health]  {resp.status_code}  {resp.json()}\n")
    resp.raise_for_status()

    # --- 2. Load local inference data ------------------------------------------
    subset = load_subset()
    feature_cols = [c for c in subset.columns if c != "true_label"]

    # --- 3. Send one batch request ---------------------------------------------
    payload = {"features": subset[feature_cols].to_dict(orient="records")}
    resp = httpx.post(f"{BASE_URL}/predict", json=payload)
    resp.raise_for_status()
    results = resp.json()

    # --- 4. Print results table ------------------------------------------------
    print(f"\n{'#':<4} {'true':>6} {'pred':>6} {'p(normal)':>10} {'p(pre-fail)':>12}  label")
    print("-" * 52)

    for i, (result, (_, row)) in enumerate(zip(results, subset.iterrows())):
        true_label = int(row["true_label"]) if "true_label" in subset.columns else "?"
        print(
            f"{i:<4} {true_label:>6} {result['prediction']:>6}"
            f"  {result['probability_normal']:>9.3f}  {result['probability_pre_failure']:>11.3f}"
            f"  {result['label']}"
        )


if __name__ == "__main__":
    main()
