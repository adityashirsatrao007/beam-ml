"""
Beam Slope & Deflection — Model Comparison Script
==================================================
Loads BeamNet and BeamPINN, compares their R², MAE, and physical/kinematic
consistency on the test set, and saves a comparison plot.
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
IMG_DIR = BASE_DIR / "docs" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

CSV_FILE = DATA_DIR / "Slope and Deflection.csv"
SECTION_IDX = 2

# ── Light Theme ───────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#000000",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "text.color": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "#CCCCCC",
    "grid.alpha": 0.5,
    "legend.facecolor": "#FFFFFF",
    "legend.edgecolor": "#000000",
    "legend.labelcolor": "black",
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 12,
    "axes.titlesize": 14,
})



# ═══════════════════════════════════════════════════════════════════════
# 1. BeamNet architecture (duplicated from train.py for standalone use)
# ═══════════════════════════════════════════════════════════════════════

class BeamNet(nn.Module):
    def __init__(self, in_features: int = 6, out_features: int = 2,
                 hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_features),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════
# 2. BeamPINN architecture (duplicated from train_pinn.py)
# ═══════════════════════════════════════════════════════════════════════

class BeamPINN(nn.Module):
    def __init__(self,
                 X_mean: np.ndarray, X_std: np.ndarray,
                 y_mean: np.ndarray, y_std: np.ndarray,
                 hidden: int = 128):
        super().__init__()
        self.register_buffer("X_mean", torch.tensor(X_mean, dtype=torch.float32))
        self.register_buffer("X_std",  torch.tensor(X_std,  dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std",  torch.tensor(y_std,  dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(6, hidden),       nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 2),
        )

    def _scale_x(self, x_raw):
        return (x_raw - self.X_mean) / self.X_std

    def _unscale_y(self, y_sc):
        return y_sc * self.y_std + self.y_mean

    def forward_raw(self, x_raw):
        x_sc = self._scale_x(x_raw)
        y_sc = self.net(x_sc)
        return self._unscale_y(y_sc)

    def forward_scaled(self, x_sc):
        return self.net(x_sc)

    def forward(self, x_sc):
        return self.net(x_sc)


# ═══════════════════════════════════════════════════════════════════════
# 3. Load data
# ═══════════════════════════════════════════════════════════════════════

def load_test_data():
    df = pd.read_csv(CSV_FILE)
    df.columns = ["X", "L", "Section", "EndCondition", "_blank", "Slope", "Deflection"]
    df = df.drop(columns=["_blank"]).dropna()
    df["EC_SS"] = (df["EndCondition"] == "SS").astype(float)
    df["EC_SF"] = (df["EndCondition"] == "SF").astype(float)
    df["EC_FF"] = (df["EndCondition"] == "FF").astype(float)

    feat_cols = ["X", "L", "Section", "EC_SS", "EC_SF", "EC_FF"]
    target_cols = ["Slope", "Deflection"]

    X_raw = df[feat_cols].values.astype(np.float32)
    y_raw = df[target_cols].values.astype(np.float32)

    scaler_X = joblib.load(MODEL_DIR / "scaler_X.pkl")
    scaler_y = joblib.load(MODEL_DIR / "scaler_y.pkl")

    X_sc = scaler_X.transform(X_raw).astype(np.float32)
    y_sc = scaler_y.transform(y_raw).astype(np.float32)

    X_tv_raw, X_test_raw, X_tv_sc, X_test_sc, y_tv_raw, y_test_raw, y_tv_sc, y_test_sc = \
        train_test_split(X_raw, X_sc, y_raw, y_sc, test_size=0.15, random_state=42)

    X_tr_raw, X_val_raw, X_tr_sc, X_val_sc, y_tr_raw, y_val_raw, y_tr_sc, y_val_sc = \
        train_test_split(X_tv_raw, X_tv_sc, y_tv_raw, y_tv_sc,
                         test_size=0.15/0.85, random_state=42)

    return (X_test_raw, X_test_sc, y_test_raw, y_test_sc,
            scaler_X, scaler_y)


# ═══════════════════════════════════════════════════════════════════════
# 4. Load models
# ═══════════════════════════════════════════════════════════════════════

def load_beamnet():
    model = BeamNet()
    model.load_state_dict(torch.load(MODEL_DIR / "beamnet.pt", map_location="cpu"))
    model.eval()
    return model


def load_beampinn():
    ckpt = torch.load(MODEL_DIR / "beampinn.pt", map_location="cpu")
    model = BeamPINN(
        X_mean=np.array(ckpt["X_mean"]),
        X_std=np.array(ckpt["X_std"]),
        y_mean=np.array(ckpt["y_mean"]),
        y_std=np.array(ckpt["y_std"]),
        hidden=128,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════
# 5. Predict & evaluate
# ═══════════════════════════════════════════════════════════════════════

def predict_beamnet(model, X_sc, scaler_y):
    with torch.no_grad():
        pred_sc = model(torch.tensor(X_sc)).numpy()
    return scaler_y.inverse_transform(pred_sc)


def predict_beampinn(model, X_raw):
    with torch.no_grad():
        pred_raw = model.forward_raw(torch.tensor(X_raw)).numpy()
    return pred_raw


def compute_metrics(truth, preds, label=""):
    results = {}
    for i, name in enumerate(["Slope", "Deflection"]):
        mae = mean_absolute_error(truth[:, i], preds[:, i])
        r2 = r2_score(truth[:, i], preds[:, i])
        results[name] = {"MAE": round(float(mae), 6), "R2": round(float(r2), 6)}
    print(f"  {label}")
    for name, m in results.items():
        print(f"    {name:12s} → MAE = {m['MAE']:.6f}  |  R² = {m['R2']:.6f}")
    return results


def kinematic_residual(model, X_raw, use_pinn=True):
    Xt = torch.tensor(X_raw)
    sec = Xt[:, SECTION_IDX:SECTION_IDX+1].detach().requires_grad_(True)
    X_with_sec = torch.cat([Xt[:, :SECTION_IDX], sec, Xt[:, SECTION_IDX+1:]], dim=1)

    if use_pinn:
        pred = model.forward_raw(X_with_sec)
        slope_p = pred[:, 0:1]
        w_p = pred[:, 1:2]
    else:
        scaler_X = joblib.load(MODEL_DIR / "scaler_X.pkl")
        scaler_y = joblib.load(MODEL_DIR / "scaler_y.pkl")
        X_mean_t = torch.tensor(scaler_X.mean_, dtype=torch.float32)
        X_std_t = torch.tensor(scaler_X.scale_, dtype=torch.float32)
        y_mean_t = torch.tensor(scaler_y.mean_, dtype=torch.float32)
        y_std_t = torch.tensor(scaler_y.scale_, dtype=torch.float32)

        X_sc = (X_with_sec - X_mean_t) / X_std_t
        pred_sc = model(X_sc)
        pred = pred_sc * y_std_t + y_mean_t
        slope_p = pred[:, 0:1]
        w_p = pred[:, 1:2]

    dw_dx = torch.autograd.grad(
        w_p, sec,
        grad_outputs=torch.ones_like(w_p),
        create_graph=False
    )[0]

    residual = float((slope_p - dw_dx).abs().mean().item())
    return residual


# ═══════════════════════════════════════════════════════════════════════
# 6. Plotting
# ═══════════════════════════════════════════════════════════════════════

def plot_comparison(X_test_raw, y_test_raw, net_preds, pinn_preds):
    targets = ["Slope", "Deflection"]
    colors = {"BeamNet": "#FF6B6B", "BeamPINN": "#00E676"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for i, target in enumerate(targets):
        truth = y_test_raw[:, i]
        n_pred = net_preds[:, i]
        p_pred = pinn_preds[:, i]

        # ── Scatter: predicted vs actual ────────────────────────────
        ax = axes[i, 0]
        ax.scatter(truth, n_pred, s=3, alpha=0.3, color=colors["BeamNet"],
                   label=f"BeamNet  (R²={r2_score(truth, n_pred):.4f})")
        ax.scatter(truth, p_pred, s=3, alpha=0.3, color=colors["BeamPINN"],
                   label=f"BeamPINN (R²={r2_score(truth, p_pred):.4f})")
        lims = [min(truth.min(), n_pred.min(), p_pred.min()),
                max(truth.max(), n_pred.max(), p_pred.max())]
        ax.plot(lims, lims, "--", color="#555555", lw=1, alpha=0.6)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{target} — Predicted vs True")
        ax.legend(markerscale=4)
        ax.set_aspect("equal")
        ax.grid(True)

        # ── Residuals ───────────────────────────────────────────────
        ax = axes[i, 1]
        n_res = truth - n_pred
        p_res = truth - p_pred
        ax.scatter(truth, n_res, s=3, alpha=0.3, color=colors["BeamNet"],
                   label="BeamNet")
        ax.scatter(truth, p_res, s=3, alpha=0.3, color=colors["BeamPINN"],
                   label="BeamPINN")
        ax.axhline(0, color="#555555", lw=1, ls="--", alpha=0.6)
        ax.set_xlabel("True")
        ax.set_ylabel("Residual")
        ax.set_title(f"{target} — Residuals")
        ax.legend(markerscale=4)
        ax.grid(True)

        # ── Error histogram ─────────────────────────────────────────
        ax = axes[i, 2]
        bins = 60
        ax.hist(n_res, bins=bins, alpha=0.5, color=colors["BeamNet"],
                label=f"BeamNet  (σ={n_res.std():.4f})")
        ax.hist(p_res, bins=bins, alpha=0.5, color=colors["BeamPINN"],
                label=f"BeamPINN (σ={p_res.std():.4f})")
        ax.set_xlabel("Error")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{target} — Error Distribution")
        ax.legend()
        ax.grid(True)

    fig.suptitle("BeamNet vs BeamPINN — Model Comparison",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.savefig(IMG_DIR / "comparison.png", bbox_inches="tight",
                facecolor="#FFFFFF")
    plt.close(fig)
    print(f"\n  Comparison plot saved → docs/images/comparison.png")



# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  BEAM MODEL COMPARISON — BeamNet vs BeamPINN")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    # ── Load data ────────────────────────────────────────────────────
    print("[1/4] Loading test data …")
    X_test_raw, X_test_sc, y_test_raw, y_test_sc, scaler_X, scaler_y = load_test_data()
    print(f"    Test samples: {len(X_test_raw):,}")

    # ── Load models ──────────────────────────────────────────────────
    print("\n[2/4] Loading models …")
    net_model = load_beamnet()
    pinn_model = load_beampinn()
    print("    BeamNet  → loaded")
    print("    BeamPINN → loaded")

    # ── Evaluate ─────────────────────────────────────────────────────
    print("\n[3/4] Evaluating …")

    # Scaled predictions for BeamNet
    net_preds = predict_beamnet(net_model, X_test_sc, scaler_y)
    # Raw predictions for BeamPINN
    pinn_preds = predict_beampinn(pinn_model, X_test_raw)

    print("\n── Metrics ─────────────────────────────────────────────")
    net_results = compute_metrics(y_test_raw, net_preds, label="BeamNet:")
    pinn_results = compute_metrics(y_test_raw, pinn_preds, label="BeamPINN:")

    # ── Kinematic residual ──────────────────────────────────────────
    print("\n── Physics (Kinematic) Consistency ──────────────────────")
    net_kin = kinematic_residual(net_model, X_test_raw, use_pinn=False)
    pinn_kin = kinematic_residual(pinn_model, X_test_raw, use_pinn=True)
    print(f"  BeamNet  |slope - dw/dx| : {net_kin:.6f}")
    print(f"  BeamPINN |slope - dw/dx| : {pinn_kin:.6f}")

    # ── Save summary ────────────────────────────────────────────────
    summary = {
        "BeamNet": {
            "metrics": net_results,
            "kinematic_residual": round(net_kin, 6),
        },
        "BeamPINN": {
            "metrics": pinn_results,
            "kinematic_residual": round(pinn_kin, 6),
        },
    }
    with open(OUTPUT_DIR / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n    Summary saved → outputs/comparison_summary.json")

    # ── Plot ─────────────────────────────────────────────────────────
    print("\n[4/4] Generating comparison plots …")
    plot_comparison(X_test_raw, y_test_raw, net_preds, pinn_preds)

    print("\n" + "=" * 65)
    print("  COMPARISON COMPLETE")
    print("=" * 65)

    # Summary table
    print(f"\n  {'':>12} {'BeamNet':>12} {'BeamPINN':>12}")
    print(f"  {'─'*38}")
    for name in ["Slope", "Deflection"]:
        print(f"  {name+'(R²)':>12} {net_results[name]['R2']:>12.4f} {pinn_results[name]['R2']:>12.4f}")
        print(f"  {name+'(MAE)':>12} {net_results[name]['MAE']:>12.6f} {pinn_results[name]['MAE']:>12.6f}")
    print(f"  {'Kinematic':>12} {net_kin:>12.6f} {pinn_kin:>12.6f}")


if __name__ == "__main__":
    main()
