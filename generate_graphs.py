"""
Beam ML — Individual Graph Generator
======================================
Every function saves exactly one standalone plot per file.
No subplot grids. White background, 200 DPI.
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

warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
IMG_DIR    = BASE_DIR / "docs" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
CSV_FILE   = DATA_DIR / "Slope and Deflection.csv"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#333333",
    "axes.titlecolor": "#111111", "text.color": "#111111",
    "xtick.color": "#555555", "ytick.color": "#555555",
    "grid.color": "#dddddd", "grid.alpha": 0.6,
    "legend.facecolor": "white", "legend.edgecolor": "#cccccc",
    "legend.labelcolor": "#333333",
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
})

C = {"bn": "#57cda4", "pn": "#ff9f0a", "tr": "#57cda4", "va": "#ff9f0a",
     "da": "#57cda4", "ki": "#5e5ce6", "bc": "#ff9f0a", "mo": "#ff375f",
     "to": "#333333", "ref": "#aaaaaa"}

class BeamNet(nn.Module):
    def __init__(self, in_features=6, out_features=2, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.BatchNorm1d(hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, out_features))
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
            nn.Linear(hidden, hidden//2), nn.Tanh(),
            nn.Linear(hidden//2, 2))
    def forward_raw(self, x_raw):
        return self.net((x_raw - self.X_mean) / self.X_std) * self.y_std + self.y_mean
    def forward(self, x_sc): return self.net(x_sc)

def load_all_data():
    df = pd.read_csv(CSV_FILE)
    df.columns = ["X","L","Section","EndCondition","_blank","Slope","Deflection"]
    df = df.drop(columns=["_blank"]).dropna()
    for c, v in [("EC_SS","SS"),("EC_SF","SF"),("EC_FF","FF")]:
        df[c] = (df["EndCondition"] == v).astype(float)
    feat_c = ["X","L","Section","EC_SS","EC_SF","EC_FF"]
    Xr = df[feat_c].values.astype(np.float32)
    yr = df[["Slope","Deflection"]].values.astype(np.float32)
    sX = joblib.load(MODEL_DIR / "scaler_X.pkl")
    sy = joblib.load(MODEL_DIR / "scaler_y.pkl")
    Xs = sX.transform(Xr).astype(np.float32)
    ys = sy.transform(yr).astype(np.float32)
    X_tvr, X_tsr, X_tvs, X_tss, y_tvr, y_tsr, y_tvs, y_tss = \
        train_test_split(Xr, Xs, yr, ys, test_size=0.15, random_state=42)
    X_trr, X_vr, X_trs, X_vs, y_trr, y_vr, y_trs, y_vs = \
        train_test_split(X_tvr, X_tvs, y_tvr, y_tvs, test_size=0.15/0.85, random_state=42)
    return X_trr, X_trs, X_vr, X_vs, X_tsr, X_tss, y_tsr, y_tss, sX, sy

def load_models():
    nm = BeamNet()
    nm.load_state_dict(torch.load(MODEL_DIR/"beamnet.pt", map_location="cpu"))
    nm.eval()
    ck = torch.load(MODEL_DIR/"beampinn.pt", map_location="cpu")
    pm = BeamPINN(X_mean=np.array(ck["X_mean"]), X_std=np.array(ck["X_std"]),
                  y_mean=np.array(ck["y_mean"]), y_std=np.array(ck["y_std"]))
    pm.load_state_dict(ck["state_dict"])
    pm.eval()
    return nm, pm

def get_preds(nm, pm, X_tsr, X_tss, sy):
    with torch.no_grad():
        np_sc = nm(torch.tensor(X_tss)).numpy()
        np_r = sy.inverse_transform(np_sc)
        pp_r = pm.forward_raw(torch.tensor(X_tsr)).numpy()
    return np_r, pp_r

def _fig(w=8, h=5):
    return plt.subplots(figsize=(w, h))

def _save(fig, name):
    fig.savefig(IMG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → docs/images/{name}")

# ═══════════════════════════ Training Curves ═══════════════════════════

def plot_beamnet_train():
    h = json.load(open(OUTPUT_DIR/"training_history.json"))
    eps = list(range(1, len(h["train_loss"])+1))
    fig, ax = _fig()
    ax.plot(eps, h["train_loss"], color=C["tr"], lw=2, label="Train MSE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("BeamNet — Train MSE"); ax.legend(); ax.grid(True)
    _save(fig, "beamnet_train_loss.png")

def plot_beamnet_val():
    h = json.load(open(OUTPUT_DIR/"training_history.json"))
    eps = list(range(1, len(h["val_loss"])+1))
    fig, ax = _fig()
    ax.plot(eps, h["val_loss"], color=C["va"], lw=2, label="Val MSE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("BeamNet — Validation MSE"); ax.legend(); ax.grid(True)
    _save(fig, "beamnet_val_loss.png")

def plot_beampinn_total():
    h = json.load(open(OUTPUT_DIR/"pinn_training_history.json"))
    eps = list(range(1, len(h["total"])+1))
    fig, ax = _fig()
    ax.plot(eps, h["total"], color=C["to"], lw=2, label="Total Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Total Loss")
    ax.set_title("BeamPINN — Total Loss"); ax.legend(); ax.grid(True)
    _save(fig, "beampinn_total_loss.png")

def plot_beampinn_components():
    h = json.load(open(OUTPUT_DIR/"pinn_training_history.json"))
    eps = list(range(1, len(h["total"])+1))
    fig, ax = _fig()
    for key, clr, lbl in [("data",C["da"],"Data"),("kinematic",C["ki"],"Kinematic"),
                           ("bc",C["bc"],"BC"),("moment",C["mo"],"Moment")]:
        ax.plot(eps, h[key], color=clr, lw=1.5, label=lbl)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("BeamPINN — Loss Components"); ax.legend(); ax.grid(True)
    _save(fig, "beampinn_loss_components.png")

def plot_beampinn_val():
    h = json.load(open(OUTPUT_DIR/"pinn_training_history.json"))
    eps = list(range(1, len(h["val_mse"])+1))
    fig, ax = _fig()
    ax.plot(eps, h["val_mse"], color=C["va"], lw=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val MSE (scaled)")
    ax.set_title("BeamPINN — Validation MSE"); ax.grid(True)
    _save(fig, "beampinn_val_mse.png")

def plot_beampinn_loss_pie():
    h = json.load(open(OUTPUT_DIR/"pinn_training_history.json"))
    final = {k: h[k][-1] for k in ("data","kinematic","bc","moment")}
    fig, ax = _fig()
    vals = list(final.values())
    lbls = [f"Data ({vals[0]:.5f})", f"Kinematic ({vals[1]:.5f})",
            f"BC ({vals[2]:.5f})", f"Moment ({vals[3]:.5f})"]
    clrs = [C["da"], C["ki"], C["bc"], C["mo"]]
    ax.pie(vals, labels=lbls, colors=clrs, autopct="%1.1f%%")
    ax.set_title("BeamPINN — Final Loss Breakdown")
    _save(fig, "beampinn_loss_pie.png")

def plot_beampinn_loss_log():
    h = json.load(open(OUTPUT_DIR/"pinn_training_history.json"))
    eps = list(range(1, len(h["total"])+1))
    fig, ax = _fig()
    for key, clr, lbl in [("data",C["da"],"Data"),("kinematic",C["ki"],"Kinematic"),
                           ("bc",C["bc"],"BC"),("moment",C["mo"],"Moment")]:
        ax.plot(eps, h[key], color=clr, lw=1.5, label=lbl)
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.set_title("BeamPINN — Loss Component Trajectory"); ax.legend(); ax.grid(True)
    _save(fig, "beampinn_loss_trajectory.png")

# ═══════════════════════════ Prediction Scatter / Residuals / Hist ════════════════

def plot_model_prediction(prefix, preds, yt, color):
    targets = ["Slope", "Deflection"]
    for i, tgt in enumerate(targets):
        t, p = yt[:, i], preds[:, i]
        mae, r2 = mean_absolute_error(t, p), r2_score(t, p)
        res = t - p; lims = [min(t.min(), p.min()), max(t.max(), p.max())]

        fig, ax = _fig()
        ax.scatter(t, p, s=4, alpha=0.3, color=color)
        ax.plot(lims, lims, "--", color=C["ref"], lw=1, alpha=0.6)
        ax.set_xlabel(f"True {tgt}"); ax.set_ylabel(f"Predicted {tgt}")
        ax.set_title(f"{prefix} — {tgt}: Predicted vs True")
        ax.text(0.05, 0.95, f"R² = {r2:.4f}\nMAE = {mae:.6f}",
                transform=ax.transAxes, va="top", color="#111",
                bbox=dict(fc="#f5f5f5", ec="#ccc", alpha=0.9))
        ax.set_aspect("equal"); ax.grid(True)
        _save(fig, f"{prefix.lower()}_{tgt.lower()}_scatter.png")

        fig, ax = _fig()
        ax.scatter(t, res, s=4, alpha=0.3, color=color)
        ax.axhline(0, color=C["ref"], lw=1, ls="--", alpha=0.6)
        ax.set_xlabel(f"True {tgt}"); ax.set_ylabel("Residual")
        ax.set_title(f"{prefix} — {tgt}: Residuals vs True"); ax.grid(True)
        _save(fig, f"{prefix.lower()}_{tgt.lower()}_residuals.png")

        fig, ax = _fig()
        ax.hist(res, bins=60, alpha=0.7, color=color, edgecolor="none")
        ax.axvline(0, color=C["ref"], lw=1, ls="--", alpha=0.6)
        ax.set_xlabel("Error"); ax.set_ylabel("Frequency")
        ax.set_title(f"{prefix} — {tgt}: Error Dist (σ={res.std():.4f})")
        ax.grid(True)
        _save(fig, f"{prefix.lower()}_{tgt.lower()}_error_hist.png")

# ═══════════════════════════ Metrics Comparisons ═══════════════════════════

def plot_r2_comparison(np_r, pp_r, yts):
    targets = ["Slope", "Deflection"]
    fig, ax = _fig()
    x = np.arange(len(targets)); w = 0.3
    bn = [round(r2_score(yts[:,i], np_r[:,i]), 4) for i in range(2)]
    pn = [round(r2_score(yts[:,i], pp_r[:,i]), 4) for i in range(2)]
    ax.bar(x-w/2, bn, w, color=C["bn"], label="BeamNet")
    ax.bar(x+w/2, pn, w, color=C["pn"], label="BeamPINN")
    ax.set_xticks(x); ax.set_xticklabels(targets)
    ax.set_ylabel("R² Score"); ax.set_title("Model Comparison — R²")
    ax.legend(); ax.set_ylim(0.98, 1.001); ax.grid(True, axis="y")
    _save(fig, "comparison_r2.png")

def plot_mae_comparison(np_r, pp_r, yts):
    targets = ["Slope", "Deflection"]
    fig, ax = _fig()
    x = np.arange(len(targets)); w = 0.3
    bn = [round(mean_absolute_error(yts[:,i], np_r[:,i]), 4) for i in range(2)]
    pn = [round(mean_absolute_error(yts[:,i], pp_r[:,i]), 4) for i in range(2)]
    ax.bar(x-w/2, bn, w, color=C["bn"], label="BeamNet")
    ax.bar(x+w/2, pn, w, color=C["pn"], label="BeamPINN")
    ax.set_xticks(x); ax.set_xticklabels(targets)
    ax.set_ylabel("MAE"); ax.set_title("Model Comparison — MAE")
    ax.legend(); ax.grid(True, axis="y")
    _save(fig, "comparison_mae.png")

def plot_kinematic_comparison():
    fig, ax = _fig()
    ax.bar(["BeamNet", "BeamPINN"], [0.086322, 0.006552],
           color=[C["bn"], C["pn"]], width=0.4)
    ax.set_ylabel("|slope − dw/dx|"); ax.set_title("Model Comparison — Kinematic Residual")
    ax.grid(True, axis="y")
    _save(fig, "comparison_kinematic.png")

# ═══════════════════════════ End Condition Performance ═══════════════════════════

def plot_end_condition_all(X_tsr, yts, np_r, pp_r):
    bc_names = ["SS (Simply Supported)", "SF (Propped Cantilever)", "FF (Fixed-Fixed)"]
    bc_col = np.argmax(X_tsr[:, 3:6], axis=1)
    for model_name, preds, color, tag in [("BeamNet", np_r, C["bn"], "beamnet"),
                                            ("BeamPINN", pp_r, C["pn"], "beampinn")]:
        for col_idx, bc_val in enumerate([0, 1, 2]):
            mask = bc_col == bc_val
            fig, ax = _fig()
            for t_idx, tgt in enumerate(["Slope", "Deflection"]):
                t, p = yts[mask, t_idx], preds[mask, t_idx]
                r2 = r2_score(t, p)
                ax.scatter(t, p, s=4, alpha=0.3, label=f"{tgt} (R²={r2:.4f})", color=color)
            lims = [yts[mask].min(), yts[mask].max()]
            ax.plot(lims, lims, "--", color=C["ref"], lw=1, alpha=0.6)
            ax.set_xlabel("True"); ax.set_ylabel("Predicted")
            ax.set_title(f"{model_name} — {bc_names[col_idx]}")
            ax.legend(); ax.set_aspect("equal"); ax.grid(True)
            tag2 = bc_names[col_idx][:2].lower()
            _save(fig, f"{tag}_{tag2}_performance.png")

# ═══════════════════════════ Data Distribution ═══════════════════════════

def plot_data_histograms():
    df = pd.read_csv(CSV_FILE)
    df.columns = ["X","L","Section","EndCondition","_blank","Slope","Deflection"]
    df = df.drop(columns=["_blank"]).dropna()
    for feat, ttl, fn in [("X","Load Position (X)","hist_x"),("L","Beam Length (L)","hist_l"),
                           ("Section","Section Position","hist_section"),
                           ("Slope","Slope Distribution","hist_slope"),
                           ("Deflection","Deflection Distribution","hist_deflection")]:
        fig, ax = _fig()
        ax.hist(df[feat], bins=50, alpha=0.7, color=C["bn"], edgecolor="none")
        ax.set_xlabel(ttl); ax.set_ylabel("Frequency"); ax.set_title(ttl); ax.grid(True)
        _save(fig, f"{fn}.png")

def plot_end_condition_pie():
    df = pd.read_csv(CSV_FILE)
    df.columns = ["X","L","Section","EndCondition","_blank","Slope","Deflection"]
    df = df.drop(columns=["_blank"]).dropna()
    fig, ax = _fig()
    bc = df["EndCondition"].value_counts()
    ax.pie(bc.values, labels=bc.index, autopct="%1.1f%%",
           colors=["#57cda4","#ff9f0a","#5e5ce6"])
    ax.set_title("End Condition Distribution")
    _save(fig, "pie_end_condition.png")

# ═══════════════════════════ Diagnostics ═══════════════════════════

def plot_residuals_vs_predicted(np_r, pp_r, yts):
    for model_name, preds, color, tag in [("BeamNet", np_r, C["bn"], "beamnet"),
                                            ("BeamPINN", pp_r, C["pn"], "beampinn")]:
        for i, tgt in enumerate(["Slope", "Deflection"]):
            fig, ax = _fig()
            res = yts[:, i] - preds[:, i]
            ax.scatter(preds[:, i], res, s=4, alpha=0.3, color=color)
            ax.axhline(0, color=C["ref"], lw=1, ls="--", alpha=0.6)
            ax.set_xlabel(f"Predicted {tgt}"); ax.set_ylabel("Residual")
            ax.set_title(f"{model_name} — {tgt}: Residuals vs Predicted"); ax.grid(True)
            _save(fig, f"{tag}_{tgt.lower()}_residuals_vs_pred.png")

def plot_qq(np_r, pp_r, yts):
    from scipy import stats as sps
    for model_name, preds, color, tag in [("BeamNet", np_r, C["bn"], "beamnet"),
                                            ("BeamPINN", pp_r, C["pn"], "beampinn")]:
        for i, tgt in enumerate(["Slope", "Deflection"]):
            fig, ax = _fig()
            res = yts[:, i] - preds[:, i]
            sr = (res - res.mean()) / res.std()
            (osm, osr), (sl, ic, _) = sps.probplot(sr, dist="norm", plot=None)
            ax.scatter(osm, osr, s=4, alpha=0.4, color=color)
            ax.plot(osm, sl*osm+ic, "--", color=C["ref"], lw=1.5, alpha=0.7)
            ax.set_xlabel("Theoretical Quantiles"); ax.set_ylabel("Sample Quantiles")
            ax.set_title(f"{model_name} — {tgt}: Q-Q Plot"); ax.grid(True)
            _, pv = sps.shapiro(res[:5000])
            ax.text(0.05, 0.95, f"Shapiro p = {pv:.2e}",
                    transform=ax.transAxes, va="top", color="#111",
                    bbox=dict(fc="#f5f5f5", ec="#ccc", alpha=0.9))
            _save(fig, f"{tag}_{tgt.lower()}_qq.png")

def plot_cumulative_error(np_r, pp_r, yts):
    for i, tgt in enumerate(["Slope", "Deflection"]):
        fig, ax = _fig()
        for name, preds, color in [("BeamNet", np_r, C["bn"]), ("BeamPINN", pp_r, C["pn"])]:
            ep = np.abs((yts[:,i] - preds[:,i]) / (yts[:,i] + 1e-12)) * 100
            th = np.linspace(0, 10, 200)
            cv = [np.mean(ep <= t)*100 for t in th]
            ax.plot(th, cv, color=color, lw=2, label=name)
        ax.set_xlabel("Absolute % Error"); ax.set_ylabel("Predictions Within (%)")
        ax.set_title(f"{tgt}: Cumulative Error Curve")
        ax.legend(); ax.set_xlim(0, 10); ax.set_ylim(0, 105); ax.grid(True)
        _save(fig, f"cumulative_error_{tgt.lower()}.png")

def plot_error_stats_bars(np_r, pp_r, yts):
    from scipy import stats as sps
    rows = []
    for i, tgt in enumerate(["Slope", "Deflection"]):
        for name, preds in [("BeamNet", np_r), ("BeamPINN", pp_r)]:
            res = yts[:,i] - preds[:,i]
            rows.append({"Model":name,"Target":tgt,"Skewness":sps.skew(res),"Kurtosis":sps.kurtosis(res)})
    df = pd.DataFrame(rows)
    for stat, ttl, fn in [("Skewness","Residual Skewness","skewness"),
                           ("Kurtosis","Residual Kurtosis","kurtosis")]:
        fig, ax = _fig()
        tgts = ["Slope", "Deflection"]; x = np.arange(len(tgts)); w = 0.3
        vn = [df.loc[(df.Model=="BeamNet")&(df.Target==t), stat].values[0] for t in tgts]
        vp = [df.loc[(df.Model=="BeamPINN")&(df.Target==t), stat].values[0] for t in tgts]
        ax.bar(x-w/2, vn, w, color=C["bn"], label="BeamNet")
        ax.bar(x+w/2, vp, w, color=C["pn"], label="BeamPINN")
        ax.axhline(0, color=C["ref"], ls="--", lw=1, alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels(tgts)
        ax.set_title(ttl); ax.legend(); ax.grid(True, axis="y")
        _save(fig, f"error_{fn}.png")

def plot_interval_coverage(np_r, pp_r, yts):
    for i, tgt in enumerate(["Slope", "Deflection"]):
        fig, ax = _fig()
        sigmas = [1, 2, 3]; x = np.arange(3); w = 0.3
        for m_idx, (name, preds, color) in enumerate(
            [("BeamNet", np_r, C["bn"]), ("BeamPINN", pp_r, C["pn"])]):
            res = yts[:,i] - preds[:,i]; std = res.std()
            cv = [np.mean(np.abs(res) <= s*std)*100 for s in sigmas]
            off = (m_idx-0.5)*w
            bars = ax.bar(x+off, cv, w, color=color, label=name, alpha=0.85)
            for b, c in zip(bars, cv):
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+1,
                        f"{c:.1f}%", ha="center", fontsize=8, color="#111")
        for s, lb in [("1σ","68.27%"),("2σ","95.45%"),("3σ","99.73%")]:
            pass
        ax.axhline(68.27, color=C["ref"], ls="--", lw=0.8, alpha=0.4)
        ax.axhline(95.45, color=C["ref"], ls="--", lw=0.8, alpha=0.4)
        ax.axhline(99.73, color=C["ref"], ls="--", lw=0.8, alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels(["1σ (68.27%)","2σ (95.45%)","3σ (99.73%)"])
        ax.set_ylabel("Actual Coverage (%)"); ax.set_title(f"{tgt}: Prediction Interval Coverage")
        ax.legend(fontsize=9); ax.set_ylim(50, 102); ax.grid(True, axis="y")
        _save(fig, f"interval_coverage_{tgt.lower()}.png")

def plot_feature_sensitivity(X_tsr, X_tss, X_tsrf, nm, pm, sy):
    feat_names = ["X (Load Pos.)","L (Beam Length)","Section #",
                  "EC = SS","EC = SF","EC = FF"]
    targets = ["Slope", "Deflection"]
    np.random.seed(42)
    idxs = np.random.choice(len(X_tss), 500, replace=False)
    Xb_sc = X_tss[idxs]; Xb_raw = X_tsrf[idxs]
    nm.eval(); pm.eval()
    with torch.no_grad():
        for f_idx in range(6):
            pert = np.linspace(-2, 2, 31)
            for m_idx, (name, model, color, tag) in enumerate(
                [("BeamNet", nm, C["bn"], "beamnet"), ("BeamPINN", pm, C["pn"], "beampinn")]):
                preds = np.zeros((len(pert), 500, 2))
                for pidx, delta in enumerate(pert):
                    Xp_sc = Xb_sc.copy(); Xp_sc[:, f_idx] += delta
                    if name == "BeamPINN":
                        Xp_rw = Xb_raw.copy()
                        Xp_rw[:, f_idx] += delta * X_tss[:, f_idx].std()
                        out = pm.forward_raw(torch.tensor(Xp_rw, dtype=torch.float32)).numpy()
                    else:
                        o_sc = nm(torch.tensor(Xp_sc, dtype=torch.float32)).numpy()
                        out = sy.inverse_transform(o_sc)
                    preds[pidx] = out
                for t_idx, tgt in enumerate(targets):
                    fig, ax = _fig()
                    mp = preds[:, :, t_idx].mean(axis=1)
                    sp = preds[:, :, t_idx].std(axis=1)
                    ax.plot(pert, mp, color=color, lw=2)
                    ax.fill_between(pert, mp-sp, mp+sp, color=color, alpha=0.15)
                    ax.set_xlabel(f"Perturbation ({feat_names[f_idx]}) [σ]")
                    ax.set_ylabel(f"Predicted {tgt}")
                    ax.set_title(f"{name} — {tgt} Sensitivity to {feat_names[f_idx]}")
                    ax.grid(True)
                    fn = feat_names[f_idx].split("(")[0].strip().lower().replace(" ","_").replace("#","no")
                    _save(fig, f"{tag}_{tgt.lower()}_sens_{fn}.png")

# ═══════════════════════════ MAIN ═══════════════════════════

def main():
    print("="*55); print("  BEAM ML — INDIVIDUAL GRAPH GENERATOR"); print("="*55)

    print("\n[1/4] Loading data & models …")
    X_trr, X_trs, X_vr, X_vs, X_tsr, X_tss, y_tsr, y_tss, sX, sy = load_all_data()
    nm, pm = load_models()
    np_r, pp_r = get_preds(nm, pm, X_tsr, X_tss, sy)
    print(f"    Test samples: {len(X_tsr):,}")

    print("\n[2/4] Training curves …")
    plot_beamnet_train(); plot_beamnet_val()
    plot_beampinn_total(); plot_beampinn_components(); plot_beampinn_val()
    plot_beampinn_loss_pie(); plot_beampinn_loss_log()

    print("\n[3/4] Prediction & comparison plots …")
    plot_model_prediction("BeamNet", np_r, y_tsr, C["bn"])
    plot_model_prediction("BeamPINN", pp_r, y_tsr, C["pn"])
    plot_r2_comparison(np_r, pp_r, y_tsr)
    plot_mae_comparison(np_r, pp_r, y_tsr)
    plot_kinematic_comparison()
    plot_end_condition_all(X_tsr, y_tsr, np_r, pp_r)
    plot_data_histograms(); plot_end_condition_pie()

    print("\n[4/4] Regression diagnostics …")
    plot_residuals_vs_predicted(np_r, pp_r, y_tsr)
    plot_qq(np_r, pp_r, y_tsr)
    plot_cumulative_error(np_r, pp_r, y_tsr)
    plot_error_stats_bars(np_r, pp_r, y_tsr)
    plot_interval_coverage(np_r, pp_r, y_tsr)
    plot_feature_sensitivity(X_tsr, X_tss, X_tsr, nm, pm, sy)

    print("\n"+"="*55); print("  ALL INDIVIDUAL GRAPHS → docs/images/"); print("="*55)

if __name__ == "__main__":
    main()
