"""
Beam Slope & Deflection — Physics-Informed Neural Network (PINN)
================================================================
Upgrades BeamNet with three physics constraints baked into the loss:

  1. KINEMATIC LOSS    : slope = dw/dx  (via autograd)
  2. BC LOSS           : w=0 at supports, θ=0 at fixed ends  (collocation)
  3. MOMENT LOSS       : d(slope)/dx = M(x)/EI  (SS beams, EI=1)

Architecture change: ReLU → Tanh (smooth, infinitely differentiable)
Scaling built into model as buffers → autograd flows through scaling correctly.

Verified BC conventions (from data inspection):
  SS : pinned-pinned   → w(0)=0, w(L)=0
  FF : fixed-fixed     → w(0)=0, w(L)=0, θ(0)=0, θ(L)=0
  SF : pinned-fixed    → w(0)=0, w(L)=0, θ(L)=0
"""

import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
MODEL_DIR  = Path(__file__).parent / "models"
OUTPUT_DIR = Path(__file__).parent / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_FILE = DATA_DIR / "Slope and Deflection.csv"

# ── Constants ─────────────────────────────────────────────────────────────
SECTION_IDX = 2          # column index of Section in feature vector
X_IDX       = 0          # column index of load position X
L_VALUE     = 5.0        # beam length (fixed in this dataset)
EI          = 1.0        # normalised flexural rigidity
N_BC_POINTS = 512        # collocation points per boundary per BC type

# ── Loss weights ─────────────────────────────────────────────────────────
LAMBDA_KIN    = 0.5      # kinematic: slope = dw/dx
LAMBDA_BC     = 0.3      # boundary conditions
LAMBDA_MOMENT = 0.1      # Euler-Bernoulli moment equation (SS only)


# ═══════════════════════════════════════════════════════════════════════════
# 1. MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

class BeamPINN(nn.Module):
    """
    Physics-Informed Neural Network for single-span beam analysis.

    Key design decisions:
    ─────────────────────
    • Tanh activations: infinitely smooth (ReLU has zero curvature,
      making d²w/dx² = 0 almost everywhere — useless for physics).
    • Scaling inside the model: StandardScaler stats stored as
      non-trainable buffers so autograd flows through scaling.
    • Two outputs: [slope, deflection] in original physical units.

    Forward modes:
    ─────────────
    • forward_raw(x_raw)    : raw inputs  → raw outputs  [used for physics loss]
    • forward_scaled(x_sc)  : scaled inputs → scaled outputs [used for data loss]
    """

    def __init__(self,
                 X_mean: np.ndarray, X_std: np.ndarray,
                 y_mean: np.ndarray, y_std: np.ndarray,
                 hidden: int = 128):
        super().__init__()

        # Scaling parameters stored as buffers (not trainable)
        self.register_buffer("X_mean", torch.tensor(X_mean, dtype=torch.float32))
        self.register_buffer("X_std",  torch.tensor(X_std,  dtype=torch.float32))
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std",  torch.tensor(y_std,  dtype=torch.float32))

        # Tanh network — smooth and infinitely differentiable
        self.net = nn.Sequential(
            nn.Linear(6, hidden),       nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 2),  # [slope, deflection]
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _scale_x(self, x_raw):
        return (x_raw - self.X_mean) / self.X_std

    def _unscale_y(self, y_sc):
        return y_sc * self.y_std + self.y_mean

    # ── Public forward passes ─────────────────────────────────────────────

    def forward_raw(self, x_raw):
        """Raw physical inputs → raw physical outputs. Use for physics loss."""
        x_sc = self._scale_x(x_raw)
        y_sc = self.net(x_sc)
        return self._unscale_y(y_sc)         # [slope_raw, w_raw]

    def forward_scaled(self, x_sc):
        """Scaled inputs → scaled outputs. Use for data loss (faster)."""
        return self.net(x_sc)

    def forward(self, x_sc):
        """Default: scaled inputs → scaled outputs."""
        return self.net(x_sc)


# ═══════════════════════════════════════════════════════════════════════════
# 2. PHYSICS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def ss_moment(x_section: torch.Tensor,
              x_load: torch.Tensor,
              L: float = L_VALUE) -> torch.Tensor:
    """
    Bending moment M(x) for a simply-supported beam
    with a unit point load P=1 at position 'x_load'.

    M(x) = (L - a)/L * x     for x ≤ a
    M(x) = a/L * (L - x)     for x ≥ a

    where a = x_load.
    """
    a  = x_load
    RA = (L - a) / L                           # reaction at left support
    m_left  = RA * x_section                   # moment for x ≤ a
    m_right = a / L * (L - x_section)          # moment for x ≥ a
    # smooth blend using heaviside step
    left_of_load = (x_section <= a).float()
    return left_of_load * m_left + (1.0 - left_of_load) * m_right


def make_bc_collocation(n: int, device: torch.device,
                        L: float = L_VALUE) -> dict:
    """
    Generate collocation points (x_load, section) at beam boundaries.
    The load position is sampled randomly across [0, L].
    Section is fixed to 0 or L (boundary).

    Returns dict with keys 'SS', 'FF', 'SF', each containing tensors
    for the boundary inputs.
    """
    # Random load positions
    x_loads = torch.rand(n, device=device) * L            # shape [n]

    def bc_input(section_val, ec_ss, ec_sf, ec_ff):
        """Build raw feature tensor for a boundary collocation point."""
        S   = torch.full((n, 1), section_val,  device=device)
        L_t = torch.full((n, 1), L,            device=device)
        X_t = x_loads.unsqueeze(1)
        EC  = torch.tensor([[ec_ss, ec_sf, ec_ff]], device=device,
                            dtype=torch.float32).expand(n, -1)
        return torch.cat([X_t, L_t, S, EC], dim=1)        # [n, 6]

    return {
        # SS: deflection=0 at both ends
        "SS_left" : bc_input(0.0,  1, 0, 0),
        "SS_right": bc_input(L,    1, 0, 0),

        # FF: deflection=0 AND slope=0 at both ends
        "FF_left" : bc_input(0.0,  0, 0, 1),
        "FF_right": bc_input(L,    0, 0, 1),

        # SF (pinned-left, fixed-right):
        #   w(0)=0  [no slope constraint at left]
        #   w(L)=0  AND  θ(L)=0  [fixed at right]
        "SF_left" : bc_input(0.0,  0, 1, 0),
        "SF_right": bc_input(L,    0, 1, 0),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. PINN LOSS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def pinn_loss(model: BeamPINN,
              X_sc_batch: torch.Tensor,
              y_sc_batch: torch.Tensor,
              X_raw_batch: torch.Tensor,
              bc_points: dict,
              lambda_kin: float = LAMBDA_KIN,
              lambda_bc:  float = LAMBDA_BC,
              lambda_mom: float = LAMBDA_MOMENT) -> tuple:
    """
    Total PINN loss = data_loss + λ_kin * kinematic_loss
                    + λ_bc  * bc_loss
                    + λ_mom * moment_loss

    Parameters
    ----------
    model        : BeamPINN
    X_sc_batch   : scaled inputs  [N, 6]
    y_sc_batch   : scaled targets [N, 2]
    X_raw_batch  : raw inputs     [N, 6]  — for autograd
    bc_points    : dict from make_bc_collocation()
    """

    # ── DATA LOSS ─────────────────────────────────────────────────────────
    pred_sc   = model.forward_scaled(X_sc_batch)
    data_loss = F.mse_loss(pred_sc, y_sc_batch)

    # ── KINEMATIC LOSS: slope = dw/dx via autograd ────────────────────────
    # We differentiate the deflection output w.r.t. the Section input.
    # Because scaling is inside the model, autograd handles chain rule.
    section_raw = X_raw_batch[:, SECTION_IDX:SECTION_IDX+1].detach().requires_grad_(True)
    X_raw_for_grad = torch.cat([
        X_raw_batch[:, :SECTION_IDX],
        section_raw,
        X_raw_batch[:, SECTION_IDX+1:]
    ], dim=1)

    pred_raw   = model.forward_raw(X_raw_for_grad)     # [N, 2] physical units
    slope_raw  = pred_raw[:, 0:1]
    w_raw      = pred_raw[:, 1:2]

    dw_dx = torch.autograd.grad(
        w_raw, section_raw,
        grad_outputs=torch.ones_like(w_raw),
        create_graph=True,
        retain_graph=True,
    )[0]                                                 # [N, 1]

    kinematic_loss = F.mse_loss(slope_raw, dw_dx)

    # ── BC LOSS: enforce deflection & slope at boundaries ────────────────
    bc_loss = torch.tensor(0.0, device=X_sc_batch.device)

    # SS boundaries — deflection = 0
    for key in ("SS_left", "SS_right"):
        pts = bc_points[key].detach().requires_grad_(False)
        pred_bc = model.forward_raw(pts)
        bc_loss = bc_loss + F.mse_loss(pred_bc[:, 1], torch.zeros(pts.shape[0], device=pts.device))

    # FF boundaries — deflection = 0 AND slope = 0
    for key in ("FF_left", "FF_right"):
        sec_ff = bc_points[key][:, SECTION_IDX:SECTION_IDX+1].detach().requires_grad_(True)
        pts_ff = torch.cat([
            bc_points[key][:, :SECTION_IDX],
            sec_ff,
            bc_points[key][:, SECTION_IDX+1:]
        ], dim=1)
        pred_ff = model.forward_raw(pts_ff)
        w_ff    = pred_ff[:, 1:2]
        dw_ff   = torch.autograd.grad(
            w_ff, sec_ff,
            grad_outputs=torch.ones_like(w_ff),
            create_graph=True, retain_graph=True
        )[0]
        bc_loss = (bc_loss
                   + F.mse_loss(pred_ff[:, 1], torch.zeros(pts_ff.shape[0], device=pts_ff.device))
                   + F.mse_loss(dw_ff.squeeze(), torch.zeros(pts_ff.shape[0], device=pts_ff.device)))

    # SF boundaries:
    #   left (x=0): deflection = 0 only   [pinned end]
    #   right (x=L): deflection = 0 AND slope = 0   [fixed end]
    pts_sf_l = bc_points["SF_left"].detach()
    pred_sf_l = model.forward_raw(pts_sf_l)
    bc_loss = bc_loss + F.mse_loss(pred_sf_l[:, 1], torch.zeros(pts_sf_l.shape[0], device=pts_sf_l.device))

    sec_sf_r = bc_points["SF_right"][:, SECTION_IDX:SECTION_IDX+1].detach().requires_grad_(True)
    pts_sf_r = torch.cat([
        bc_points["SF_right"][:, :SECTION_IDX],
        sec_sf_r,
        bc_points["SF_right"][:, SECTION_IDX+1:]
    ], dim=1)
    pred_sf_r = model.forward_raw(pts_sf_r)
    w_sf_r    = pred_sf_r[:, 1:2]
    dw_sf_r   = torch.autograd.grad(
        w_sf_r, sec_sf_r,
        grad_outputs=torch.ones_like(w_sf_r),
        create_graph=True, retain_graph=True
    )[0]
    bc_loss = (bc_loss
               + F.mse_loss(pred_sf_r[:, 1], torch.zeros(pts_sf_r.shape[0], device=pts_sf_r.device))
               + F.mse_loss(dw_sf_r.squeeze(), torch.zeros(pts_sf_r.shape[0], device=pts_sf_r.device)))

    # ── MOMENT LOSS: d(slope)/dx = M(x)/EI for SS beams ─────────────────
    # Uses second-order autograd on Section input.
    ss_mask = X_raw_batch[:, 3] == 1.0                  # EC_SS column
    moment_loss = torch.tensor(0.0, device=X_sc_batch.device)

    if ss_mask.sum() > 10:
        X_ss     = X_raw_batch[ss_mask]
        sec_ss   = X_ss[:, SECTION_IDX:SECTION_IDX+1].detach().requires_grad_(True)
        X_ss_cat = torch.cat([X_ss[:, :SECTION_IDX], sec_ss, X_ss[:, SECTION_IDX+1:]], dim=1)

        pred_ss  = model.forward_raw(X_ss_cat)
        slope_ss = pred_ss[:, 0:1]

        # First derivative: d(slope)/dx
        d_slope_dx = torch.autograd.grad(
            slope_ss, sec_ss,
            grad_outputs=torch.ones_like(slope_ss),
            create_graph=True, retain_graph=True
        )[0]

        # Analytical moment M(x) for SS beam
        x_load_ss = X_ss[:, X_IDX:X_IDX+1]
        M_x = ss_moment(sec_ss, x_load_ss, L=L_VALUE)

        # Euler-Bernoulli: d(slope)/dx = M(x)/EI = M(x)  [EI=1]
        moment_loss = F.mse_loss(d_slope_dx, M_x.detach())

    # ── TOTAL LOSS ────────────────────────────────────────────────────────
    total = (data_loss
             + lambda_kin  * kinematic_loss
             + lambda_bc   * bc_loss
             + lambda_mom  * moment_loss)

    return total, data_loss, kinematic_loss, bc_loss, moment_loss


# ═══════════════════════════════════════════════════════════════════════════
# 4. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_and_prepare():
    """Load CSV, encode BCs, scale, split. Returns raw + scaled tensors."""
    print(f"[1/5] Loading data …")
    df = pd.read_csv(CSV_FILE)
    df.columns = ["X", "L", "Section", "EndCondition", "_blank", "Slope", "Deflection"]
    df = df.drop(columns=["_blank"]).dropna()

    df["EC_SS"] = (df["EndCondition"] == "SS").astype(float)
    df["EC_SF"] = (df["EndCondition"] == "SF").astype(float)
    df["EC_FF"] = (df["EndCondition"] == "FF").astype(float)

    feat_cols   = ["X", "L", "Section", "EC_SS", "EC_SF", "EC_FF"]
    target_cols = ["Slope", "Deflection"]

    X_raw = df[feat_cols].values.astype(np.float32)
    y_raw = df[target_cols].values.astype(np.float32)

    print(f"    Rows: {len(df):,} | BCs: {df['EndCondition'].value_counts().to_dict()}")

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_sc     = scaler_X.fit_transform(X_raw).astype(np.float32)
    y_sc     = scaler_y.fit_transform(y_raw).astype(np.float32)

    X_tv_raw, X_test_raw, X_tv_sc, X_test_sc, y_tv_raw, y_test_raw, y_tv_sc, y_test_sc = \
        train_test_split(X_raw, X_sc, y_raw, y_sc, test_size=0.15, random_state=42)

    X_tr_raw, X_val_raw, X_tr_sc, X_val_sc, y_tr_raw, y_val_raw, y_tr_sc, y_val_sc = \
        train_test_split(X_tv_raw, X_tv_sc, y_tv_raw, y_tv_sc,
                         test_size=0.15/0.85, random_state=42)

    print(f"    Train: {len(X_tr_raw):,} | Val: {len(X_val_raw):,} | Test: {len(X_test_raw):,}")
    return (X_tr_raw, X_tr_sc, y_tr_raw, y_tr_sc,
            X_val_raw, X_val_sc, y_val_raw, y_val_sc,
            X_test_raw, X_test_sc, y_test_raw, y_test_sc,
            scaler_X, scaler_y)


# ═══════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════

def train(model, X_tr_raw, X_tr_sc, y_tr_sc,
          X_val_sc, y_val_sc,
          epochs: int = 200, batch_size: int = 512, lr: float = 8e-4,
          device: torch.device = torch.device("cpu")):

    Xr_t = torch.tensor(X_tr_raw).to(device)
    Xs_t = torch.tensor(X_tr_sc).to(device)
    ys_t = torch.tensor(y_tr_sc).to(device)
    Xv_t = torch.tensor(X_val_sc).to(device)
    yv_t = torch.tensor(y_val_sc).to(device)

    loader = DataLoader(TensorDataset(Xr_t, Xs_t, ys_t),
                        batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val, best_state = float("inf"), None
    history = {k: [] for k in ("total","data","kinematic","bc","moment","val_mse")}

    hdr = f"{'Ep':>5} {'Total':>9} {'Data':>9} {'Kinem':>9} {'BC':>9} {'Moment':>9} {'ValMSE':>9}"
    print(f"\n{hdr}\n{'─'*65}")

    for ep in range(1, epochs + 1):
        model.train()
        ep_totals = {k: 0.0 for k in ("total","data","kinematic","bc","moment")}
        ep_n = 0

        # Refresh boundary collocation points each epoch
        bc_pts = make_bc_collocation(N_BC_POINTS, device)

        for Xr_b, Xs_b, ys_b in loader:
            optimizer.zero_grad()
            total, data, kin, bc, mom = pinn_loss(
                model, Xs_b, ys_b, Xr_b, bc_pts
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n = len(Xs_b)
            ep_totals["total"]     += total.item() * n
            ep_totals["data"]      += data.item()  * n
            ep_totals["kinematic"] += kin.item()   * n
            ep_totals["bc"]        += bc.item()    * n
            ep_totals["moment"]    += mom.item()   * n
            ep_n += n

        scheduler.step()

        # Validation (data loss only, fast)
        model.eval()
        with torch.no_grad():
            val_mse = F.mse_loss(model.forward_scaled(Xv_t), yv_t).item()

        for k in history:
            if k == "val_mse":
                history[k].append(val_mse)
            else:
                history[k].append(ep_totals[k] / ep_n)

        if val_mse < best_val:
            best_val  = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 20 == 0 or ep == 1:
            print(f"{ep:>5} "
                  f"{ep_totals['total']/ep_n:>9.5f} "
                  f"{ep_totals['data']/ep_n:>9.5f} "
                  f"{ep_totals['kinematic']/ep_n:>9.5f} "
                  f"{ep_totals['bc']/ep_n:>9.5f} "
                  f"{ep_totals['moment']/ep_n:>9.5f} "
                  f"{val_mse:>9.5f}")

    model.load_state_dict(best_state)
    return model, history


# ═══════════════════════════════════════════════════════════════════════════
# 6. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test_raw, y_test_raw, device, scaler_y):
    model.eval()
    Xt = torch.tensor(X_test_raw).to(device)
    with torch.no_grad():
        pred_raw = model.forward_raw(Xt).cpu().numpy()

    results = {}
    for i, name in enumerate(["Slope", "Deflection"]):
        mae = mean_absolute_error(y_test_raw[:, i], pred_raw[:, i])
        r2  = r2_score(y_test_raw[:, i], pred_raw[:, i])
        results[name] = {"MAE": round(float(mae), 6), "R2": round(float(r2), 6)}
        print(f"    {name:12s} → MAE = {mae:.6f}  |  R² = {r2:.6f}")
    return results, pred_raw


def evaluate_physics(model, X_test_raw, device):
    """Check how well physics constraints are satisfied on test set."""
    model.eval()
    Xt = torch.tensor(X_test_raw).to(device)

    sec = Xt[:, SECTION_IDX:SECTION_IDX+1].detach().requires_grad_(True)
    X_with_sec = torch.cat([Xt[:, :SECTION_IDX], sec, Xt[:, SECTION_IDX+1:]], dim=1)
    pred = model.forward_raw(X_with_sec)
    slope_p = pred[:, 0:1]
    w_p     = pred[:, 1:2]

    dw_dx = torch.autograd.grad(
        w_p, sec,
        grad_outputs=torch.ones_like(w_p),
        create_graph=False
    )[0]

    kin_residual = (slope_p - dw_dx).abs().mean().item()
    print(f"    Kinematic residual |slope - dw/dx| : {kin_residual:.6f}")
    return kin_residual


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  BEAM PINN — PHYSICS-INFORMED NEURAL NETWORK TRAINING")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    # ── 1. Data ──────────────────────────────────────────────────────────
    (X_tr_raw, X_tr_sc, y_tr_raw, y_tr_sc,
     X_val_raw, X_val_sc, y_val_raw, y_val_sc,
     X_test_raw, X_test_sc, y_test_raw, y_test_sc,
     scaler_X, scaler_y) = load_and_prepare()

    # ── 2. Build model ───────────────────────────────────────────────────
    print("\n[2/5] Building BeamPINN …")
    model = BeamPINN(
        X_mean=scaler_X.mean_,  X_std=scaler_X.scale_,
        y_mean=scaler_y.mean_,  y_std=scaler_y.scale_,
        hidden=128
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Trainable parameters: {n_params:,}")

    # ── 3. Train ─────────────────────────────────────────────────────────
    print("\n[3/5] Training …")
    model, history = train(
        model,
        X_tr_raw, X_tr_sc, y_tr_sc,
        X_val_sc, y_val_sc,
        epochs=200, batch_size=512, lr=8e-4, device=device
    )

    # ── 4. Save ───────────────────────────────────────────────────────────
    print("\n[4/5] Saving …")
    torch.save({
        "state_dict": model.state_dict(),
        "X_mean": scaler_X.mean_.tolist(),
        "X_std":  scaler_X.scale_.tolist(),
        "y_mean": scaler_y.mean_.tolist(),
        "y_std":  scaler_y.scale_.tolist(),
    }, MODEL_DIR / "beampinn.pt")
    with open(OUTPUT_DIR / "pinn_training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("    Saved → models/beampinn.pt")

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    print("\n[5/5] Evaluation …")
    results, preds = evaluate_model(model, X_test_raw, y_test_raw, device, scaler_y)

    print("\n  Physics constraint verification:")
    kin_res = evaluate_physics(model, X_test_raw, device)

    summary = {
        "model":      "BeamPINN (PyTorch)",
        "train_rows": len(X_tr_raw),
        "test_rows":  len(X_test_raw),
        "physics_losses": {
            "lambda_kinematic": LAMBDA_KIN,
            "lambda_bc":        LAMBDA_BC,
            "lambda_moment":    LAMBDA_MOMENT,
        },
        "metrics":          results,
        "kinematic_residual": round(kin_res, 6),
    }
    with open(OUTPUT_DIR / "pinn_evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 65)
    print("  PINN TRAINING COMPLETE")
    print(f"  Slope      R² = {results['Slope']['R2']:.4f}")
    print(f"  Deflection R² = {results['Deflection']['R2']:.4f}")
    print(f"  Physics kinematic residual = {kin_res:.6f}")
    print(f"  Saved → outputs/pinn_evaluation_summary.json")
    print("=" * 65)


if __name__ == "__main__":
    main()
