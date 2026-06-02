"""
Beam Slope & Deflection — Inference / Prediction Script
========================================================
Supports both BeamNet (standard ML) and BeamPINN (physics-informed).
Default model: BeamPINN.

Usage:
    python predict.py
    or import and call predict_beam(X=2.5, L=5.0, Section=1.5,
                                     end_condition="SS", model_type="beampinn")
"""

import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

MODEL_DIR = Path(__file__).parent / "models"

# Load scalers (always needed for BeamNet & preprocessing)
scaler_X = joblib.load(MODEL_DIR / "scaler_X.pkl")
scaler_y = joblib.load(MODEL_DIR / "scaler_y.pkl")

FEATURE_COLS = ["X", "L", "Section", "EC_SS", "EC_SF", "EC_FF"]
EC_MAP = {"SS": [1, 0, 0], "SF": [0, 1, 0], "FF": [0, 0, 1]}

import torch
import torch.nn as nn


# ── Model architectures (inline, no imports from train modules) ─────────

class BeamNet(nn.Module):
    def __init__(self, in_features=6, out_features=2, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, out_features),
        )

    def forward(self, x):
        return self.net(x)


class BeamPINN(nn.Module):
    def __init__(self, X_mean, X_std, y_mean, y_std, hidden=128):
        super().__init__()
        self.register_buffer("X_mean", torch.tensor(X_mean, dtype=torch.float32))
        self.register_buffer("X_std",  torch.tensor(X_std,  dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std",  torch.tensor(y_std,  dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 2),
        )

    def forward_raw(self, x_raw):
        return (self.net((x_raw - self.X_mean) / self.X_std) * self.y_std + self.y_mean)

    def forward(self, x_sc):
        return self.net(x_sc)


# ── Model loading (lazy) ───────────────────────────────────────────────

_models = {}

def _load_model(model_type: str):
    if model_type in _models:
        return _models[model_type]

    if model_type == "beamnet":
        model = BeamNet()
        model.load_state_dict(torch.load(MODEL_DIR / "beamnet.pt", map_location="cpu"))
        model.eval()
    elif model_type == "beampinn":
        ckpt = torch.load(MODEL_DIR / "beampinn.pt", map_location="cpu")
        model = BeamPINN(
            X_mean=np.array(ckpt["X_mean"]),
            X_std=np.array(ckpt["X_std"]),
            y_mean=np.array(ckpt["y_mean"]),
            y_std=np.array(ckpt["y_std"]),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'beamnet' or 'beampinn'.")

    _models[model_type] = model
    return model


def predict_beam(X: float, L: float, Section: float,
                 end_condition: str, model_type: str = "beampinn") -> dict:
    """
    Predict slope and deflection for a beam configuration.

    Parameters
    ----------
    X              : float — Position of the unit load along the beam (0 to L)
    L              : float — Total length of the beam
    Section        : float — Cross-section position where output is evaluated
    end_condition  : str   — "SS", "SF", or "FF"
    model_type     : str   — "beampinn" (default, physics-informed) or "beamnet"

    Returns
    -------
    dict with keys "Slope" and "Deflection"
    """
    if end_condition not in EC_MAP:
        raise ValueError(f"end_condition must be one of {list(EC_MAP)}")

    ec = EC_MAP[end_condition]
    raw = np.array([[X, L, Section, *ec]], dtype=np.float32)
    model = _load_model(model_type)

    with torch.no_grad():
        if model_type == "beampinn":
            pred = model.forward_raw(torch.tensor(raw)).numpy()[0]
        else:
            scaled = scaler_X.transform(raw).astype(np.float32)
            pred_sc = model(torch.tensor(scaled)).numpy()
            pred = scaler_y.inverse_transform(pred_sc)[0]

    return {"Slope": float(pred[0]), "Deflection": float(pred[1])}


def predict_both(X: float, L: float, Section: float,
                 end_condition: str) -> dict:
    """Return predictions from both models for side-by-side comparison."""
    return {
        "BeamNet":  predict_beam(X, L, Section, end_condition, "beamnet"),
        "BeamPINN": predict_beam(X, L, Section, end_condition, "beampinn"),
    }


if __name__ == "__main__":
    test_cases = [
        {"X": 2.5, "L": 5.0, "Section": 2.5, "end_condition": "SS"},
        {"X": 1.0, "L": 5.0, "Section": 3.0, "end_condition": "SF"},
        {"X": 3.0, "L": 5.0, "Section": 1.5, "end_condition": "FF"},
    ]

    HEADER_FMT = "  {:<45} {:>12} {:>12}  {:>12} {:>12}"
    ROW_FMT    = "  {:<45} {:>12.6f} {:>12.6f}  {:>12.6f} {:>12.6f}"

    print("=" * 100)
    print("  BEAM SLOPE & DEFLECTION — PREDICTION DEMO (Both Models)")
    print("=" * 100)
    print(HEADER_FMT.format(
        "Case", "Slope(Net)", "Defl(Net)", "Slope(PINN)", "Defl(PINN)"
    ))
    print("  " + "─" * 95)

    for case in test_cases:
        both = predict_both(**case)
        label = f"X={case['X']}, L={case['L']}, Sec={case['Section']}, BC={case['end_condition']}"
        print(ROW_FMT.format(
            label,
            both["BeamNet"]["Slope"], both["BeamNet"]["Deflection"],
            both["BeamPINN"]["Slope"], both["BeamPINN"]["Deflection"],
        ))

    print("=" * 100)
