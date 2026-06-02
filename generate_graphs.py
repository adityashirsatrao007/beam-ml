"""
Beam ML — Comprehensive Graph Generator
=========================================
Generates individual high-quality plots for the beam slope & deflection project.
Saves all to docs/images/ using Apple Dark theme (200 DPI).
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

# ── Paths ──
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
MODEL_DIR   = BASE_DIR / "models"
OUTPUT_DIR  = BASE_DIR / "outputs"
IMG_DIR     = BASE_DIR / "docs" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILE    = DATA_DIR / "Slope and Deflection.csv"
SECTION_IDX = 2

# ── Apple Dark Theme ──
plt.rcParams.update({
    "figure.facecolor": "#1C1C1E",
    "axes.facecolor": "#2C2C2E",
    "axes.edgecolor": "#555557",
    "axes.labelcolor": "white",
    "axes.titlecolor": "white",
    "text.color": "white",
    "xtick.color": "#aeaeb2",
    "ytick.color": "#aeaeb2",
    "grid.color": "#3a3a3c",
    "grid.alpha": 0.3,
    "legend.facecolor": "#2C2C2E",
    "legend.edgecolor": "#555557",
    "legend.labelcolor": "white",
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

COLORS = {
    "beamnet":   "#57cda4",
    "beampinn":  "#ff9f0a",
    "train":     "#57cda4",
    "val":       "#ff9f0a",
    "true":      "#aeaeb2",
    "data":      "#57cda4",
    "kinematic": "#5e5ce6",
    "bc":        "#ff9f0a",
    "moment":    "#ff375f",
    "total":     "#ffffff",
}


# ════════════════════════════════════════════════════
# Model Architectures
# ════════════════════════════════════════════════════

class BeamNet(nn.Module):
    def __init__(self, in_features=6, out_features=2, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, out_features),
        )
    def forward(self, x): return self.net(x)


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
        return self.net((x_raw - self.X_mean) / self.X_std) * self.y_std + self.y_mean
    def forward(self, x_sc): return self.net(x_sc)


# ════════════════════════════════════════════════════
# Data Loading
# ════════════════════════════════════════════════════

def load_all_data():
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
        train_test_split(X_tv_raw, X_tv_sc, y_tv_raw, y_tv_sc, test_size=0.15/0.85, random_state=42)

    return (X_tr_raw, X_tr_sc, X_val_raw, X_val_sc,
            X_test_raw, X_test_sc, y_test_raw, y_test_sc, scaler_X, scaler_y)


# ════════════════════════════════════════════════════
# Model Loading
# ════════════════════════════════════════════════════

def load_models():
    net_model = BeamNet()
    net_model.load_state_dict(torch.load(MODEL_DIR / "beamnet.pt", map_location="cpu"))
    net_model.eval()

    ckpt = torch.load(MODEL_DIR / "beampinn.pt", map_location="cpu")
    pinn_model = BeamPINN(
        X_mean=np.array(ckpt["X_mean"]), X_std=np.array(ckpt["X_std"]),
        y_mean=np.array(ckpt["y_mean"]), y_std=np.array(ckpt["y_std"]),
    )
    pinn_model.load_state_dict(ckpt["state_dict"])
    pinn_model.eval()
    return net_model, pinn_model


def get_predictions(net_model, pinn_model, X_test_raw, X_test_sc, scaler_y):
    with torch.no_grad():
        net_pred_sc = net_model(torch.tensor(X_test_sc)).numpy()
        net_preds = scaler_y.inverse_transform(net_pred_sc)
        pinn_preds = pinn_model.forward_raw(torch.tensor(X_test_raw)).numpy()
    return net_preds, pinn_preds


# ════════════════════════════════════════════════════
# Plotting Functions
# ════════════════════════════════════════════════════

def plot_beamnet_training():
    with open(OUTPUT_DIR / "training_history.json") as f:
        h = json.load(f)
    epochs = list(range(1, len(h["train_loss"]) + 1))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, h["train_loss"], color=COLORS["train"], lw=2, alpha=0.8, label="Train MSE")
    ax.plot(epochs, h["val_loss"], color=COLORS["val"], lw=2, alpha=0.8, label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("BeamNet — Training Curve")
    ax.legend()
    ax.grid(True)
    fig.savefig(IMG_DIR / "beamnet_training_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved → docs/images/beamnet_training_curve.png")


def plot_beampinn_training():
    with open(OUTPUT_DIR / "pinn_training_history.json") as f:
        h = json.load(f)
    epochs = list(range(1, len(h["total"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # All loss components
    ax = axes[0]
    losses = [
        ("total",    COLORS["total"],     "Total"),
        ("data",     COLORS["data"],      "Data (MSE)"),
        ("kinematic", COLORS["kinematic"], "Kinematic"),
        ("bc",       COLORS["bc"],        "BC"),
        ("moment",   COLORS["moment"],    "Moment"),
    ]
    for key, color, label in losses:
        ax.plot(epochs, h[key], color=color, lw=1.5, alpha=0.8, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("BeamPINN — All Loss Components")
    ax.legend(fontsize=10)
    ax.grid(True)

    # Validation MSE
    ax = axes[1]
    ax.plot(epochs, h["val_mse"], color=COLORS["val"], lw=2, alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val MSE (scaled)")
    ax.set_title("BeamPINN — Validation MSE")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "beampinn_training_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved → docs/images/beampinn_training_curves.png")


def plot_predictions(model_name, preds, y_true, color):
    targets = ["Slope", "Deflection"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for i, target in enumerate(targets):
        t = y_true[:, i]
        p = preds[:, i]
        mae = mean_absolute_error(t, p)
        r2 = r2_score(t, p)
        res = t - p
        lims = [min(t.min(), p.min()), max(t.max(), p.max())]

        # Predicted vs True
        ax = axes[i, 0]
        ax.scatter(t, p, s=4, alpha=0.3, color=color)
        ax.plot(lims, lims, "--", color="#aeaeb2", lw=1, alpha=0.6)
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_title(f"{model_name} — {target}: Predicted vs True")
        ax.text(0.05, 0.95, f"R² = {r2:.4f}\nMAE = {mae:.6f}",
                transform=ax.transAxes, va="top", color="white",
                bbox=dict(facecolor="#1C1C1E", edgecolor="#555557", alpha=0.8))
        ax.set_aspect("equal")
        ax.grid(True)

        # Residuals
        ax = axes[i, 1]
        ax.scatter(t, res, s=4, alpha=0.3, color=color)
        ax.axhline(0, color="#aeaeb2", lw=1, ls="--", alpha=0.6)
        ax.set_xlabel(f"True {target}")
        ax.set_ylabel("Residual")
        ax.set_title(f"{model_name} — {target}: Residuals")
        ax.grid(True)

        # Error histogram
        ax = axes[i, 2]
        ax.hist(res, bins=60, alpha=0.7, color=color, edgecolor="none")
        ax.axvline(0, color="#aeaeb2", lw=1, ls="--", alpha=0.6)
        ax.set_xlabel("Error")
        ax.set_ylabel("Frequency")
        ax.set_title(f"{model_name} — {target}: Error Dist (σ={res.std():.4f})")
        ax.grid(True)

    fig.tight_layout()
    tag = model_name.lower().replace(" ", "_")
    fig.savefig(IMG_DIR / f"{tag}_predictions.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → docs/images/{tag}_predictions.png")


def plot_metrics_comparison(net_preds, pinn_preds, y_test_raw):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    targets = ["Slope", "Deflection"]

    r2_data = {m: {t: [] for t in targets} for m in ["BeamNet", "BeamPINN"]}
    mae_data = {m: {t: [] for t in targets} for m in ["BeamNet", "BeamPINN"]}

    for i, target in enumerate(targets):
        r2_data["BeamNet"][target]   = round(r2_score(y_test_raw[:, i], net_preds[:, i]), 4)
        r2_data["BeamPINN"][target]  = round(r2_score(y_test_raw[:, i], pinn_preds[:, i]), 4)
        mae_data["BeamNet"][target]  = round(mean_absolute_error(y_test_raw[:, i], net_preds[:, i]), 4)
        mae_data["BeamPINN"][target] = round(mean_absolute_error(y_test_raw[:, i], pinn_preds[:, i]), 4)

    x = np.arange(len(targets))
    w = 0.3

    # R² Bar Chart
    ax = axes[0]
    ax.bar(x - w/2, [r2_data["BeamNet"][t] for t in targets], w, color=COLORS["beamnet"], label="BeamNet")
    ax.bar(x + w/2, [r2_data["BeamPINN"][t] for t in targets], w, color=COLORS["beampinn"], label="BeamPINN")
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("R² Score")
    ax.set_title("R² Comparison")
    ax.legend()
    ax.set_ylim(0.98, 1.001)
    ax.grid(True, axis="y")

    # MAE Bar Chart
    ax = axes[1]
    ax.bar(x - w/2, [mae_data["BeamNet"][t] for t in targets], w, color=COLORS["beamnet"], label="BeamNet")
    ax.bar(x + w/2, [mae_data["BeamPINN"][t] for t in targets], w, color=COLORS["beampinn"], label="BeamPINN")
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylabel("MAE")
    ax.set_title("MAE Comparison")
    ax.legend()
    ax.grid(True, axis="y")

    # Kinematic Residual
    ax = axes[2]
    net_kin = 0.086322
    pinn_kin = 0.006552
    ax.bar(["BeamNet", "BeamPINN"], [net_kin, pinn_kin],
           color=[COLORS["beamnet"], COLORS["beampinn"]], width=0.4)
    ax.set_ylabel("|slope − dw/dx|")
    ax.set_title("Kinematic Residual")
    ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(IMG_DIR / "metrics_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved → docs/images/metrics_comparison.png")


def plot_loss_breakdown():
    with open(OUTPUT_DIR / "pinn_training_history.json") as f:
        h = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = list(range(1, len(h["total"]) + 1))

    # Final epoch breakdown pie
    ax = axes[0]
    final = {k: h[k][-1] for k in ("data", "kinematic", "bc", "moment")}
    labels = [f"Data ({final['data']:.5f})", f"Kinematic ({final['kinematic']:.5f})",
              f"BC ({final['bc']:.5f})", f"Moment ({final['moment']:.5f})"]
    colors_pie = [COLORS["data"], COLORS["kinematic"], COLORS["bc"], COLORS["moment"]]
    ax.pie(final.values(), labels=labels, colors=colors_pie, autopct="%1.1f%%",
           textprops={"color": "white"})
    ax.set_title("BeamPINN — Final Loss Breakdown")

    # Loss trajectory (log scale)
    ax = axes[1]
    for key, color, label in [
        ("data", COLORS["data"], "Data"),
        ("kinematic", COLORS["kinematic"], "Kinematic"),
        ("bc", COLORS["bc"], "BC"),
        ("moment", COLORS["moment"], "Moment"),
    ]:
        ax.plot(epochs, h[key], color=color, lw=1.5, alpha=0.8, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("BeamPINN — Loss Component Trajectory")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "beampinn_loss_breakdown.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved → docs/images/beampinn_loss_breakdown.png")


def plot_end_condition_performance(X_test_raw, y_test_raw, net_preds, pinn_preds):
    bc_map = {3: "SS", 4: "SF", 5: "FF"}
    bc_names = ["SS (Simply Supported)", "SF (Propped Cantilever)", "FF (Fixed-Fixed)"]
    bc_col = np.argmax(X_test_raw[:, 3:6], axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    targets = ["Slope", "Deflection"]
    models = [("BeamNet", net_preds, COLORS["beamnet"]),
              ("BeamPINN", pinn_preds, COLORS["beampinn"])]

    for col_idx, bc_val in enumerate([0, 1, 2]):
        mask = bc_col == bc_val
        for row_idx, (name, preds, color) in enumerate(models):
            ax = axes[row_idx, col_idx]
            for t_idx, target in enumerate(targets):
                t = y_test_raw[mask, t_idx]
                p = preds[mask, t_idx]
                r2 = r2_score(t, p)
                ax.scatter(t, p, s=3, alpha=0.3, label=f"{target} (R²={r2:.4f})")
            lims = [y_test_raw[mask].min(), y_test_raw[mask].max()]
            ax.plot(lims, lims, "--", color="#aeaeb2", lw=1, alpha=0.6)
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")
            ax.set_title(f"{name} — {bc_names[col_idx]}")
            ax.legend(markerscale=4, fontsize=8)
            ax.set_aspect("equal")
            ax.grid(True)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "end_condition_performance.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved → docs/images/end_condition_performance.png")


def plot_data_distribution():
    df = pd.read_csv(CSV_FILE)
    df.columns = ["X", "L", "Section", "EndCondition", "_blank", "Slope", "Deflection"]
    df = df.drop(columns=["_blank"]).dropna()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    features = ["X", "L", "Section", "Slope", "Deflection"]
    titles = ["Load Position (X)", "Beam Length (L)", "Section Position",
              "Slope Distribution", "Deflection Distribution"]

    for i, (feat, title) in enumerate(zip(features, titles)):
        ax = axes[i // 3, i % 3]
        ax.hist(df[feat], bins=50, alpha=0.7, color=COLORS["beamnet"], edgecolor="none")
        ax.set_xlabel(title)
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.grid(True)

    # End Condition pie
    ax = axes[1, 2]
    bc_counts = df["EndCondition"].value_counts()
    ax.pie(bc_counts.values, labels=bc_counts.index, autopct="%1.1f%%",
           colors=["#57cda4", "#ff9f0a", "#5e5ce6"],
           textprops={"color": "white"})
    ax.set_title("End Condition Distribution")

    fig.tight_layout()
    fig.savefig(IMG_DIR / "data_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved → docs/images/data_distribution.png")


def plot_combined_training_curves():
    net_h = json.load(open(OUTPUT_DIR / "training_history.json"))
    pinn_h = json.load(open(OUTPUT_DIR / "pinn_training_history.json"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(list(range(1, len(net_h["train_loss"])+1)), net_h["train_loss"],
            color=COLORS["train"], lw=2, label="Train MSE")
    ax.plot(list(range(1, len(net_h["val_loss"])+1)), net_h["val_loss"],
            color=COLORS["val"], lw=2, label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("BeamNet — Train / Val Loss")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    epochs_p = list(range(1, len(pinn_h["total"])+1))
    ax.plot(epochs_p, pinn_h["total"], color=COLORS["total"], lw=2, label="Total")
    ax.plot(epochs_p, pinn_h["data"], color=COLORS["data"], lw=1.5, label="Data")
    ax.plot(epochs_p, pinn_h["val_mse"], color=COLORS["val"], lw=2, ls="--", label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("BeamPINN — PINN Loss Components")
    ax.legend(fontsize=9)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "combined_training_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved → docs/images/combined_training_curves.png")


# ════════════════════════════════════════════════════
# Regression Diagnostic Plots
# ════════════════════════════════════════════════════

def plot_residuals_vs_predicted(net_preds, pinn_preds, y_test_raw):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    targets = ["Slope", "Deflection"]
    models = [("BeamNet", net_preds, COLORS["beamnet"]),
              ("BeamPINN", pinn_preds, COLORS["beampinn"])]

    for t_idx, target in enumerate(targets):
        for m_idx, (name, preds, color) in enumerate(models):
            ax = axes[m_idx, t_idx]
            residuals = y_test_raw[:, t_idx] - preds[:, t_idx]
            ax.scatter(preds[:, t_idx], residuals, s=4, alpha=0.3, color=color)
            ax.axhline(0, color="#aeaeb2", lw=1, ls="--", alpha=0.6)
            ax.set_xlabel(f"Predicted {target}")
            ax.set_ylabel("Residual")
            ax.set_title(f"{name} — {target}: Residuals vs Predicted")
            ax.grid(True)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "residuals_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved → docs/images/residuals_vs_predicted.png")


def plot_qq_residuals(net_preds, pinn_preds, y_test_raw):
    from scipy import stats as sp_stats

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    targets = ["Slope", "Deflection"]
    models = [("BeamNet", net_preds, COLORS["beamnet"]),
              ("BeamPINN", pinn_preds, COLORS["beampinn"])]

    for t_idx, target in enumerate(targets):
        for m_idx, (name, preds, color) in enumerate(models):
            ax = axes[m_idx, t_idx]
            residuals = y_test_raw[:, t_idx] - preds[:, t_idx]

            # Standardize residuals for Q-Q
            std_resid = (residuals - residuals.mean()) / residuals.std()
            (osm, osr), (slope, intercept, r) = sp_stats.probplot(std_resid, dist="norm", plot=None)

            ax.scatter(osm, osr, s=4, alpha=0.4, color=color)
            ax.plot(osm, slope * osm + intercept, "--", color="#aeaeb2", lw=1.5, alpha=0.7)
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            ax.set_title(f"{name} — {target}: Q-Q Plot")
            ax.grid(True)

            # Annotate normality test
            stat, p_val = sp_stats.shapiro(residuals[:5000])
            ax.text(0.05, 0.95, f"Shapiro p = {p_val:.2e}",
                    transform=ax.transAxes, va="top", color="white",
                    bbox=dict(facecolor="#1C1C1E", edgecolor="#555557", alpha=0.8))

    fig.tight_layout()
    fig.savefig(IMG_DIR / "qq_residuals.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved → docs/images/qq_residuals.png")


def plot_cumulative_error_curve(net_preds, pinn_preds, y_test_raw):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    targets = ["Slope", "Deflection"]
    models = [("BeamNet", net_preds, COLORS["beamnet"]),
              ("BeamPINN", pinn_preds, COLORS["beampinn"])]
    max_err_pct = 10.0

    for t_idx, target in enumerate(targets):
        ax = axes[t_idx]
        for name, preds, color in models:
            err_pct = np.abs((y_test_raw[:, t_idx] - preds[:, t_idx]) / (y_test_raw[:, t_idx] + 1e-12)) * 100
            thresholds = np.linspace(0, max_err_pct, 200)
            coverage = [np.mean(err_pct <= t) * 100 for t in thresholds]
            ax.plot(thresholds, coverage, color=color, lw=2, label=name)

        # Annotate key thresholds
        for name, preds, color in models:
            err_pct = np.abs((y_test_raw[:, t_idx] - preds[:, t_idx]) / (y_test_raw[:, t_idx] + 1e-12)) * 100
            for thresh in [1.0, 5.0]:
                cov = np.mean(err_pct <= thresh) * 100
                ax.axvline(thresh, color=color, ls=":", alpha=0.3)
                ax.text(thresh + 0.1, 50 if name == "BeamNet" else 45,
                        f"{name} @{thresh}%: {cov:.1f}%", fontsize=8, color=color,
                        rotation=90, alpha=0.7)

        ax.set_xlabel("Absolute Percentage Error (%)")
        ax.set_ylabel("Predictions Within Threshold (%)")
        ax.set_title(f"{target}: Cumulative Error Curve")
        ax.legend()
        ax.set_xlim(0, max_err_pct)
        ax.set_ylim(0, 105)
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(IMG_DIR / "cumulative_error_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved → docs/images/cumulative_error_curves.png")


def plot_error_statistics(net_preds, pinn_preds, y_test_raw):
    from scipy import stats as sp_stats

    targets = ["Slope", "Deflection"]
    models = [("BeamNet", net_preds, COLORS["beamnet"]),
              ("BeamPINN", pinn_preds, COLORS["beampinn"])]

    rows = []
    for t_idx, target in enumerate(targets):
        for name, preds, _ in models:
            resid = y_test_raw[:, t_idx] - preds[:, t_idx]
            std_resid = (resid - resid.mean()) / resid.std()
            _, p_val = sp_stats.shapiro(resid[:5000])
            rows.append({
                "Model": name, "Target": target,
                "MAE": mean_absolute_error(y_test_raw[:, t_idx], preds[:, t_idx]),
                "RMSE": np.sqrt(np.mean(resid ** 2)),
                "MaxAE": np.max(np.abs(resid)),
                "σ(resid)": resid.std(),
                "Skewness": sp_stats.skew(resid),
                "Kurtosis": sp_stats.kurtosis(resid),
                "Shapiro p": p_val,
                "Q1": np.percentile(resid, 25),
                "Median": np.median(resid),
                "Q3": np.percentile(resid, 75),
                "IQR": np.percentile(resid, 75) - np.percentile(resid, 25),
            })

    stats_df = pd.DataFrame(rows)

    # Plot: grouped bar of skewness + kurtosis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for s_idx, (stat_name, title) in enumerate([
        ("Skewness", "Residual Skewness (0 = symmetric)"),
        ("Kurtosis", "Residual Kurtosis (0 = normal tail)"),
    ]):
        ax = axes[s_idx]
        x = np.arange(len(targets))
        w = 0.3
        vals_net = [stats_df.loc[(stats_df["Model"] == "BeamNet") & (stats_df["Target"] == t), stat_name].values[0] for t in targets]
        vals_pinn = [stats_df.loc[(stats_df["Model"] == "BeamPINN") & (stats_df["Target"] == t), stat_name].values[0] for t in targets]
        ax.bar(x - w/2, vals_net, w, color=COLORS["beamnet"], label="BeamNet")
        ax.bar(x + w/2, vals_pinn, w, color=COLORS["beampinn"], label="BeamPINN")
        ax.axhline(0, color="#aeaeb2", lw=1, ls="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(targets)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(IMG_DIR / "error_statistics.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved → docs/images/error_statistics.png")

    # Save table as JSON
    stats_df.to_json(OUTPUT_DIR / "error_statistics.json", orient="records", indent=2)
    print("  Saved → outputs/error_statistics.json")


def plot_prediction_interval_coverage(net_preds, pinn_preds, y_test_raw):
    targets = ["Slope", "Deflection"]
    models = [("BeamNet", net_preds, COLORS["beamnet"]),
              ("BeamPINN", pinn_preds, COLORS["beampinn"])]
    sigmas = [1, 2, 3]
    labels = ["1σ (68.27%)", "2σ (95.45%)", "3σ (99.73%)"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for t_idx, target in enumerate(targets):
        ax = axes[t_idx]
        x = np.arange(len(sigmas))
        w = 0.3

        for m_idx, (name, preds, color) in enumerate(models):
            resid = y_test_raw[:, t_idx] - preds[:, t_idx]
            std_res = resid.std()
            coverage = []
            for s in sigmas:
                within = np.mean(np.abs(resid) <= s * std_res) * 100
                coverage.append(within)
            offset = (m_idx - 0.5) * w
            bars = ax.bar(x + offset, coverage, w, color=color, label=name, alpha=0.85)
            for bar, cov in zip(bars, coverage):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{cov:.1f}%", ha="center", fontsize=8, color="white")

        # Theoretical reference lines
        for s, label in zip(sigmas, labels):
            ax.axhline([68.27, 95.45, 99.73][s - 1], color="#aeaeb2", ls="--", lw=0.8, alpha=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Actual Coverage (%)")
        ax.set_title(f"{target}: Prediction Interval Coverage")
        ax.legend(fontsize=9)
        ax.set_ylim(50, 102)
        ax.grid(True, axis="y")

    fig.tight_layout()
    fig.savefig(IMG_DIR / "prediction_interval_coverage.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved → docs/images/prediction_interval_coverage.png")


def plot_feature_sensitivity(X_test_raw, X_test_sc, X_test_raw_full,
                              net_model, pinn_model, scaler_y):
    feat_names = ["X (Load Pos.)", "L (Beam Length)", "Section #",
                  "EC = SS", "EC = SF", "EC = FF"]
    targets = ["Slope", "Deflection"]
    n_samples = 500
    np.random.seed(42)

    net_model.eval()
    pinn_model.eval()

    # Pick a subset of test points
    idxs = np.random.choice(len(X_test_sc), n_samples, replace=False)
    X_base_sc = X_test_sc[idxs]
    X_base_raw = X_test_raw_full[idxs]

    fig, axes = plt.subplots(2, 6, figsize=(22, 8))
    models_plot = [("BeamNet", net_model, COLORS["beamnet"]),
                   ("BeamPINN", pinn_model, COLORS["beampinn"])]

    with torch.no_grad():
        for f_idx in range(6):
            perturbations = np.linspace(-2, 2, 31)
            for m_idx, (name, model, color) in enumerate(models_plot):
                preds = np.zeros((len(perturbations), n_samples, 2))
                for p_idx, delta in enumerate(perturbations):
                    X_pert_sc = X_base_sc.copy()
                    X_pert_sc[:, f_idx] += delta
                    if name == "BeamPINN":
                        X_pert_raw = X_base_raw.copy()
                        X_pert_raw[:, f_idx] += delta * X_test_sc[:, f_idx].std()
                        out = pinn_model.forward_raw(torch.tensor(X_pert_raw, dtype=torch.float32)).numpy()
                    else:
                        out_sc = net_model(torch.tensor(X_pert_sc, dtype=torch.float32)).numpy()
                        out = scaler_y.inverse_transform(out_sc)
                    preds[p_idx] = out

                for t_idx, target in enumerate(targets):
                    ax = axes[t_idx, f_idx]
                    mean_pred = preds[:, :, t_idx].mean(axis=1)
                    std_pred = preds[:, :, t_idx].std(axis=1)
                    ax.plot(perturbations, mean_pred, color=color, lw=1.5, label=name)
                    ax.fill_between(perturbations,
                                    mean_pred - std_pred, mean_pred + std_pred,
                                    color=color, alpha=0.1)
                    ax.set_xlabel(feat_names[f_idx], fontsize=9)
                    if f_idx == 0:
                        ax.set_ylabel(target, fontsize=9)
                    if t_idx == 0:
                        ax.set_title(feat_names[f_idx], fontsize=10)
                    ax.grid(True, alpha=0.3)

    fig.suptitle("Feature Sensitivity: Predicted Output vs Perturbed Input (±2σ scaled)",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(IMG_DIR / "feature_sensitivity.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved → docs/images/feature_sensitivity.png")


# ════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  BEAM ML — GRAPH GENERATOR")
    print("=" * 55)

    print("\n[1/4] Loading data & models …")
    (X_tr_raw, X_tr_sc, X_val_raw, X_val_sc,
     X_test_raw, X_test_sc, y_test_raw, y_test_sc,
     scaler_X, scaler_y) = load_all_data()
    net_model, pinn_model = load_models()
    net_preds, pinn_preds = get_predictions(net_model, pinn_model,
                                            X_test_raw, X_test_sc, scaler_y)
    print(f"    Test samples: {len(X_test_raw):,}")

    print("\n[2/4] Generating training curve plots …")
    plot_beamnet_training()
    plot_beampinn_training()
    plot_combined_training_curves()
    plot_loss_breakdown()

    print("\n[3/4] Generating prediction & comparison plots …")
    plot_predictions("BeamNet", net_preds, y_test_raw, COLORS["beamnet"])
    plot_predictions("BeamPINN", pinn_preds, y_test_raw, COLORS["beampinn"])
    plot_metrics_comparison(net_preds, pinn_preds, y_test_raw)
    plot_end_condition_performance(X_test_raw, y_test_raw, net_preds, pinn_preds)
    plot_data_distribution()

    print("\n[4/4] Generating regression diagnostic plots …")
    plot_residuals_vs_predicted(net_preds, pinn_preds, y_test_raw)
    plot_qq_residuals(net_preds, pinn_preds, y_test_raw)
    plot_cumulative_error_curve(net_preds, pinn_preds, y_test_raw)
    plot_error_statistics(net_preds, pinn_preds, y_test_raw)
    plot_prediction_interval_coverage(net_preds, pinn_preds, y_test_raw)
    plot_feature_sensitivity(X_test_raw, X_test_sc, X_test_raw,
                              net_model, pinn_model, scaler_y)

    print("\n" + "=" * 55)
    print("  ALL GRAPHS GENERATED → docs/images/")
    print("=" * 55)


if __name__ == "__main__":
    main()
