"""
Beam Slope & Deflection — Inference / Prediction Script
========================================================
Use this after training to predict slope and deflection for new beam cases.

Usage:
    python predict.py
    or import and call predict_beam(X=2.5, L=5.0, Section=1.5, end_condition="SS")
"""

import numpy as np
import joblib
import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"

# Load scalers (always needed)
scaler_X = joblib.load(MODEL_DIR / "scaler_X.pkl")
scaler_y = joblib.load(MODEL_DIR / "scaler_y.pkl")

# Feature order must match training
FEATURE_COLS = ["X", "L", "Section", "EC_SS", "EC_SF", "EC_FF"]
EC_MAP = {"SS": [1, 0, 0], "SF": [0, 1, 0], "FF": [0, 0, 1]}

# Try loading PyTorch model first, fall back to sklearn
try:
    import torch
    from train import BeamNet  # reuses architecture definition
    model = BeamNet()
    model.load_state_dict(torch.load(MODEL_DIR / "beamnet.pt", map_location="cpu"))
    model.eval()
    USE_TORCH = True
except Exception:
    import joblib as jl
    model = jl.load(MODEL_DIR / "beamnet_sklearn.pkl")
    USE_TORCH = False


def predict_beam(X: float, L: float, Section: float, end_condition: str) -> dict:
    """
    Predict slope and deflection for a beam configuration.

    Parameters
    ----------
    X              : float — Position of the unit load along the beam (0 to L)
    L              : float — Total length of the beam
    Section        : float — Cross-section position where output is evaluated
    end_condition  : str   — "SS" (simply supported), "SF" (propped cantilever),
                              or "FF" (fixed-fixed)

    Returns
    -------
    dict with keys "Slope" and "Deflection"
    """
    if end_condition not in EC_MAP:
        raise ValueError(f"end_condition must be one of {list(EC_MAP)}")

    ec = EC_MAP[end_condition]
    raw = np.array([[X, L, Section, *ec]], dtype=np.float32)
    scaled = scaler_X.transform(raw).astype(np.float32)

    if USE_TORCH:
        import torch
        with torch.no_grad():
            pred_scaled = model(torch.tensor(scaled)).numpy()
    else:
        pred_scaled = model.predict(scaled)

    pred = scaler_y.inverse_transform(pred_scaled)[0]
    return {"Slope": float(pred[0]), "Deflection": float(pred[1])}


if __name__ == "__main__":
    # ── Demo predictions ──────────────────────────────────────────────────
    test_cases = [
        {"X": 2.5, "L": 5.0, "Section": 2.5, "end_condition": "SS"},
        {"X": 1.0, "L": 5.0, "Section": 3.0, "end_condition": "SF"},
        {"X": 3.0, "L": 5.0, "Section": 1.5, "end_condition": "FF"},
    ]

    print("=" * 55)
    print("  BEAM SLOPE & DEFLECTION — PREDICTION DEMO")
    print("=" * 55)
    for case in test_cases:
        result = predict_beam(**case)
        print(f"\n  Input  : X={case['X']}, L={case['L']}, "
              f"Section={case['Section']}, BC={case['end_condition']}")
        print(f"  Slope      = {result['Slope']:.6f}")
        print(f"  Deflection = {result['Deflection']:.6f}")
    print("=" * 55)
