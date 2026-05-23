"""
Demo Client — Inference Server
==============================
This script demonstrates how to call the FastAPI inference server end-to-end:

  1. Hit GET /health to confirm the server is running.
  2. Download a pre-engineered feature subset from GitHub.
  3. Send the entire subset as a single POST /predict batch request.
  4. Print a comparison table of predicted vs. true labels.

Usage:
  # Server must already be running in another terminal:
  #   uvicorn Project.Server.inference_server:app --reload
  python Project/Server/call_inference_server.py
"""

import os
from pathlib import Path

import httpx
import pandas as pd

# BASE_URL can be overridden via environment variable, which makes it trivial
# to point this client at a server running in Docker or on a remote machine:
#   BASE_URL=http://my-server:8000 python call_inference_server.py
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

# The inference subset is hosted on GitHub (via Git LFS).
# Using media.githubusercontent.com gives us a direct download URL that
# bypasses the LFS pointer redirect, so pandas can read it with read_csv()
# without any additional tooling. The subset is small (a few hundred rows of
# pre-computed sliding-window features) — downloading on every run is fast.
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
    # Always probe /health before sending real work. raise_for_status() turns
    # any 4xx/5xx into an exception immediately, so we get a clear error
    # message ("server not found") rather than a confusing prediction failure.
    resp = httpx.get(f"{BASE_URL}/health")
    print(f"[health]  {resp.status_code}  {resp.json()}\n")
    resp.raise_for_status()

    # --- 2. Load inference data ------------------------------------------------
    subset = load_subset()

    # `true_label` is ground truth for evaluation — it must NOT be sent to the
    # model as a feature. We drop it by listing all other column names.
    feature_cols = [c for c in subset.columns if c != "true_label"]

    # --- 3. Send one single batch request -------------------------------------
    # to_dict(orient="records") converts the DataFrame to a list of dicts,
    # one dict per row, which matches the PredictRequest.features schema:
    #   [{"feat_A": 1.2, ...}, {"feat_A": 3.1, ...}, ...]
    # Sending all rows in a single POST is much more efficient than one
    # request per row because it avoids repeated HTTP round-trip overhead
    # and lets the model score all rows in one vectorised call.
    payload = {"features": subset[feature_cols].to_dict(orient="records")}
    resp = httpx.post(f"{BASE_URL}/predict", json=payload)
    # Raise immediately if the server returned an error status (e.g. 422
    # Unprocessable Entity for a bad payload, or 500 for a server crash).
    resp.raise_for_status()
    results = resp.json()

    # --- 4. Print results table -----------------------------------------------
    # Column widths are chosen so the data rows below align cleanly.
    print(f"\n{'#':<4} {'true':>6} {'pred':>6} {'p(normal)':>10} {'p(pre-fail)':>12}  label")
    print("-" * 52)

    # zip() pairs each server response dict with the corresponding row from
    # the subset DataFrame (which holds the ground-truth label).
    for i, (result, (_, row)) in enumerate(zip(results, subset.iterrows())):
        # The guard handles subsets that were created without a true_label
        # column — the client still works, it just shows "?" in that column.
        true_label = int(row["true_label"]) if "true_label" in subset.columns else "?"
        print(
            f"{i:<4} {true_label:>6} {result['prediction']:>6}"
            f"  {result['probability_normal']:>9.3f}  {result['probability_pre_failure']:>11.3f}"
            f"  {result['label']}"
        )


if __name__ == "__main__":
    main()
