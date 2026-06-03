# -*- coding: utf-8 -*-
"""
Beam ML Pipeline: BeamNet vs Physics-Informed BeamPINN
======================================================
This file merges all training, evaluation, comparison, and plotting steps into a single,
self-contained script designed to run seamlessly in Google Colab or local Python environments.

How to Run in Google Colab:
---------------------------
1. Upload this file (`beam_ml_colab.py`) or copy its contents into a Colab cell.
2. Run the cell. If 'Slope and Deflection.csv' is not in the Colab workspace, you will be
   prompted to upload it automatically.
3. The script will train:
   - BeamNet (Standard Pure ML Network) for 150 epochs
   - BeamPINN (Physics-Informed Neural Network) for 200 epochs
4. Finally, it will generate comparative metric tables and performance graphs.
"""

# %% [markdown]
# # Step 1: Install & Import Dependencies

# %%
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# # Step 2: Handle Dataset Upload/Loading

# %%
CSV_FILENAME = "Slope and Deflection.csv"

# If file is not found, prompt for upload in Google Colab
if not os.path.exists(CSV_FILENAME):
    print(f"'{CSV_FILENAME}' not found in the current directory.")
    try:
        from google.colab import files
        print("Prompting for file upload in Google Colab...")
        uploaded = files.upload()
        for fn in uploaded.keys():
            if fn.lower().endswith('.csv'):
                os.rename(fn, CSV_FILENAME)
                print(f"Successfully uploaded and renamed {fn} to {CSV_FILENAME}")
                break
    except ImportError:
        print("Not running in Google Colab environment.")
        print(f"Please place '{CSV_FILENAME}' in the same directory as this script and run again.")
        sys.exit(1)

# %% [markdown]
# # Step 3: Data Loading & Preprocessing

# %%
# Indices for PINN calculations
X_IDX = 0
L_IDX = 1
SECTION_IDX = 2
L_VALUE = 5.0
N_BC_POINTS = 512

# Loss weights for PINN
LAMBDA_KIN    = 0.5      # kinematic: slope = dw/dx
LAMBDA_BC     = 0.3      # boundary conditions
LAMBDA_MOMENT = 0.1      # Euler-Bernoulli moment equation (SS only)

def load_and_prepare_data():
    print(f"Loading data from {CSV_FILENAME} ...")
    df = pd.read_csv(CSV_FILENAME)
    
    # Rename columns to standard names
    df.columns = ["X", "L", "Section", "EndCondition", "_blank", "Slope", "Deflection"]
    df = df.drop(columns=["_blank"]).dropna()
    
    # Encode boundary conditions
    df["EC_SS"] = (df["EndCondition"] == "SS").astype(float)
    df["EC_SF"] = (df["EndCondition"] == "SF").astype(float)
    df["EC_FF"] = (df["EndCondition"] == "FF").astype(float)
    
    feat_cols = ["X", "L", "Section", "EC_SS", "EC_SF", "EC_FF"]
    target_cols = ["Slope", "Deflection"]
    
    X_raw = df[feat_cols].values.astype(np.float32)
    y_raw = df[target_cols].values.astype(np.float32)
    
    print(f"    Rows loaded: {len(df):,}")
    print(f"    End conditions: {df['EndCondition'].value_counts().to_dict()}")
    
    # Standard scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_sc = scaler_X.fit_transform(X_raw).astype(np.float32)
    y_sc = scaler_y.fit_transform(y_raw).astype(np.float32)
    
    # Train/Val/Test Split (70% train / 15% val / 15% test)
    X_tv_raw, X_test_raw, X_tv_sc, X_test_sc, y_tv_raw, y_test_raw, y_tv_sc, y_test_sc = \
        train_test_split(X_raw, X_sc, y_raw, y_sc, test_size=0.15, random_state=42)
        
    X_tr_raw, X_val_raw, X_tr_sc, X_val_sc, y_tr_raw, y_val_raw, y_tr_sc, y_val_sc = \
        train_test_split(X_tv_raw, X_tv_sc, y_tv_raw, y_tv_sc, test_size=0.15/0.85, random_state=42)
        
    print(f"    Train size: {len(X_tr_raw):,} | Val size: {len(X_val_raw):,} | Test size: {len(X_test_raw):,}")
    return (X_tr_raw, X_tr_sc, y_tr_raw, y_tr_sc,
            X_val_raw, X_val_sc, y_val_raw, y_val_sc,
            X_test_raw, X_test_sc, y_test_raw, y_test_sc,
            scaler_X, scaler_y)

# Load data
X_tr_raw, X_tr_sc, y_tr_raw, y_tr_sc, \
X_val_raw, X_val_sc, y_val_raw, y_val_sc, \
X_test_raw, X_test_sc, y_test_raw, y_test_sc, \
scaler_X, scaler_y = load_and_prepare_data()

# %% [markdown]
# # Step 4: Model Architectures

# %%
class BeamNet(nn.Module):
    """
    Standard Feedforward Neural Network (BeamNet)
    - Input (6) -> [128 -> BN -> ReLU -> Dropout] x 3 -> Output (2)
    """
    def __init__(self, in_features: int = 6, out_features: int = 2, hidden: int = 128, dropout: float = 0.2):
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

class BeamPINN(nn.Module):
    """
    Physics-Informed Neural Network (BeamPINN)
    - Uses infinitely differentiable Tanh activations.
    - Stores mean and standard deviations internally to support autograd backpropagation.
    """
    def __init__(self, X_mean: np.ndarray, X_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, hidden: int = 128):
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

# %% [markdown]
# # Step 5: Physics Formulation and PINN Loss Functions

# %%
def ss_moment(x_section: torch.Tensor, x_load: torch.Tensor, L: float = L_VALUE) -> torch.Tensor:
    """
    Analytical Bending Moment M(x) for simply supported beam with point load.
    M(x) = (L - a)/L * x     for x <= a
    M(x) = a/L * (L - x)     for x >= a
    where a is the load position.
    """
    a = x_load
    RA = (L - a) / L
    m_left = RA * x_section
    m_right = (a / L) * (L - x_section)
    left_of_load = (x_section <= a).float()
    return left_of_load * m_left + (1.0 - left_of_load) * m_right

def make_bc_collocation(n: int, device: torch.device, L: float = L_VALUE) -> dict:
    """
    Generates boundary condition points to compute PINN boundary loss.
    """
    x_loads = torch.rand(n, device=device) * L
    
    def bc_input(section_val, ec_ss, ec_sf, ec_ff):
        S = torch.full((n, 1), section_val, device=device)
        L_t = torch.full((n, 1), L, device=device)
        X_t = x_loads.unsqueeze(1)
        EC = torch.tensor([[ec_ss, ec_sf, ec_ff]], device=device, dtype=torch.float32).expand(n, -1)
        return torch.cat([X_t, L_t, S, EC], dim=1)
        
    return {
        "SS_left" : bc_input(0.0, 1, 0, 0),
        "SS_right": bc_input(L,   1, 0, 0),
        
        "FF_left" : bc_input(0.0, 0, 0, 1),
        "FF_right": bc_input(L,   0, 0, 1),
        
        "SF_left" : bc_input(0.0, 0, 1, 0),
        "SF_right": bc_input(L,   0, 1, 0),
    }

def pinn_loss(model: BeamPINN, X_sc_batch: torch.Tensor, y_sc_batch: torch.Tensor,
              X_raw_batch: torch.Tensor, bc_points: dict) -> tuple:
    # 1. Data loss
    pred_sc = model.forward_scaled(X_sc_batch)
    data_loss = F.mse_loss(pred_sc, y_sc_batch)
    
    # 2. Kinematic Loss (slope = dw/dx)
    sec_raw = X_raw_batch[:, SECTION_IDX:SECTION_IDX+1].detach().requires_grad_(True)
    X_raw_for_grad = torch.cat([
        X_raw_batch[:, :SECTION_IDX],
        sec_raw,
        X_raw_batch[:, SECTION_IDX+1:]
    ], dim=1)
    
    pred_raw = model.forward_raw(X_raw_for_grad)
    slope_raw = pred_raw[:, 0:1]
    w_raw = pred_raw[:, 1:2]
    
    dw_dx = torch.autograd.grad(
        w_raw, sec_raw,
        grad_outputs=torch.ones_like(w_raw),
        create_graph=True,
        retain_graph=True
    )[0]
    
    kin_loss = F.mse_loss(slope_raw, dw_dx)
    
    # 3. Boundary Condition Loss
    bc_loss = torch.tensor(0.0, device=X_sc_batch.device)
    
    # SS boundaries (w = 0)
    for key in ("SS_left", "SS_right"):
        pts = bc_points[key].detach().requires_grad_(False)
        pred_bc = model.forward_raw(pts)
        bc_loss = bc_loss + F.mse_loss(pred_bc[:, 1], torch.zeros(pts.shape[0], device=pts.device))
        
    # FF boundaries (w = 0 and dw/dx = 0)
    for key in ("FF_left", "FF_right"):
        sec_ff = bc_points[key][:, SECTION_IDX:SECTION_IDX+1].detach().requires_grad_(True)
        pts_ff = torch.cat([
            bc_points[key][:, :SECTION_IDX],
            sec_ff,
            bc_points[key][:, SECTION_IDX+1:]
        ], dim=1)
        pred_ff = model.forward_raw(pts_ff)
        w_ff = pred_ff[:, 1:2]
        dw_ff = torch.autograd.grad(
            w_ff, sec_ff,
            grad_outputs=torch.ones_like(w_ff),
            create_graph=True, retain_graph=True
        )[0]
        bc_loss = bc_loss + F.mse_loss(pred_ff[:, 1], torch.zeros_like(w_ff)) + F.mse_loss(dw_ff, torch.zeros_like(dw_ff))
        
    # SF boundaries (left x=0 pinned: w=0; right x=L fixed: w=0 and dw/dx=0)
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
    w_sf_r = pred_sf_r[:, 1:2]
    dw_sf_r = torch.autograd.grad(
        w_sf_r, sec_sf_r,
        grad_outputs=torch.ones_like(w_sf_r),
        create_graph=True, retain_graph=True
    )[0]
    bc_loss = bc_loss + F.mse_loss(pred_sf_r[:, 1], torch.zeros_like(w_sf_r)) + F.mse_loss(dw_sf_r, torch.zeros_like(dw_sf_r))
    
    # 4. Moment Loss (d(slope)/dx = M(x) for simply supported beam)
    ss_mask = X_raw_batch[:, 3] == 1.0
    moment_loss = torch.tensor(0.0, device=X_sc_batch.device)
    
    if ss_mask.sum() > 10:
        X_ss = X_raw_batch[ss_mask]
        sec_ss = X_ss[:, SECTION_IDX:SECTION_IDX+1].detach().requires_grad_(True)
        X_ss_cat = torch.cat([X_ss[:, :SECTION_IDX], sec_ss, X_ss[:, SECTION_IDX+1:]], dim=1)
        
        pred_ss = model.forward_raw(X_ss_cat)
        slope_ss = pred_ss[:, 0:1]
        
        d_slope_dx = torch.autograd.grad(
            slope_ss, sec_ss,
            grad_outputs=torch.ones_like(slope_ss),
            create_graph=True, retain_graph=True
        )[0]
        
        x_load_ss = X_ss[:, X_IDX:X_IDX+1]
        M_x = ss_moment(sec_ss, x_load_ss, L=L_VALUE)
        moment_loss = F.mse_loss(d_slope_dx, M_x.detach())
        
    total_loss = data_loss + LAMBDA_KIN * kin_loss + LAMBDA_BC * bc_loss + LAMBDA_MOMENT * moment_loss
    return total_loss, data_loss, kin_loss, bc_loss, moment_loss

# %% [markdown]
# # Step 6: Train Standard Model (BeamNet) — 150 Epochs

# %%
print("\n=== Training BeamNet (Standard Pure ML Network) ===")
print("Training for 150 epochs...")

net_model = BeamNet().to(device)
net_optimizer = torch.optim.Adam(net_model.parameters(), lr=1e-3, weight_decay=1e-5)
net_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(net_optimizer, patience=10, factor=0.5, min_lr=1e-6)
loss_fn = nn.MSELoss()

# Prepare datasets
Xt_tr = torch.tensor(X_tr_sc).to(device)
yt_tr = torch.tensor(y_tr_sc).to(device)
Xt_val = torch.tensor(X_val_sc).to(device)
yt_val = torch.tensor(y_val_sc).to(device)

net_loader = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=512, shuffle=True)

best_net_val = float("inf")
best_net_state = None

print(f"{'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}  {'LR':>10}")
print("─" * 45)

for epoch in range(1, 150 + 1):
    net_model.train()
    epoch_loss = 0.0
    for Xb, yb in net_loader:
        net_optimizer.zero_grad()
        loss = loss_fn(net_model(Xb), yb)
        loss.backward()
        net_optimizer.step()
        epoch_loss += loss.item() * len(Xb)
    train_mse = epoch_loss / len(Xt_tr)
    
    net_model.eval()
    with torch.no_grad():
        val_mse = loss_fn(net_model(Xt_val), yt_val).item()
        
    net_scheduler.step(val_mse)
    
    if val_mse < best_net_val:
        best_net_val = val_mse
        best_net_state = {k: v.cpu().clone() for k, v in net_model.state_dict().items()}
        
    if epoch % 15 == 0 or epoch == 1:
        lr_now = net_optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6}  {train_mse:>12.6f}  {val_mse:>12.6f}  {lr_now:>10.2e}")

net_model.load_state_dict(best_net_state)
print("BeamNet Training Complete!")

# %% [markdown]
# # Step 7: Train Physics-Informed Model (BeamPINN) — 200 Epochs

# %%
print("\n=== Training BeamPINN (Physics-Informed Network) ===")
print("Training for 200 epochs...")

pinn_model = BeamPINN(
    X_mean=scaler_X.mean_, X_std=scaler_X.scale_,
    y_mean=scaler_y.mean_, y_std=scaler_y.scale_,
    hidden=128
).to(device)

pinn_optimizer = torch.optim.Adam(pinn_model.parameters(), lr=8e-4, weight_decay=1e-5)
pinn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pinn_optimizer, T_max=200)

pinn_loader = DataLoader(
    TensorDataset(torch.tensor(X_tr_raw).to(device), torch.tensor(X_tr_sc).to(device), torch.tensor(y_tr_sc).to(device)),
    batch_size=512, shuffle=True
)

best_pinn_val = float("inf")
best_pinn_state = None

print(f"{'Ep':>5} {'Total':>9} {'Data':>9} {'Kinem':>9} {'BC':>9} {'Moment':>9} {'ValMSE':>9}")
print("─" * 65)

for ep in range(1, 200 + 1):
    pinn_model.train()
    ep_totals = {k: 0.0 for k in ("total", "data", "kinematic", "bc", "moment")}
    ep_n = 0
    
    # Boundary points sampled every epoch
    bc_pts = make_bc_collocation(N_BC_POINTS, device)
    
    for Xr_b, Xs_b, ys_b in pinn_loader:
        pinn_optimizer.zero_grad()
        total, data, kin, bc, mom = pinn_loss(pinn_model, Xs_b, ys_b, Xr_b, bc_pts)
        total.backward()
        torch.nn.utils.clip_grad_norm_(pinn_model.parameters(), 1.0)
        pinn_optimizer.step()
        
        n = len(Xs_b)
        ep_totals["total"] += total.item() * n
        ep_totals["data"] += data.item() * n
        ep_totals["kinematic"] += kin.item() * n
        ep_totals["bc"] += bc.item() * n
        ep_totals["moment"] += mom.item() * n
        ep_n += n
        
    pinn_scheduler.step()
    
    pinn_model.eval()
    with torch.no_grad():
        val_mse = F.mse_loss(pinn_model.forward_scaled(torch.tensor(X_val_sc).to(device)), torch.tensor(y_val_sc).to(device)).item()
        
    if val_mse < best_pinn_val:
        best_pinn_val = val_mse
        best_pinn_state = {k: v.cpu().clone() for k, v in pinn_model.state_dict().items()}
        
    if ep % 20 == 0 or ep == 1:
        print(f"{ep:>5} "
              f"{ep_totals['total']/ep_n:>9.5f} "
              f"{ep_totals['data']/ep_n:>9.5f} "
              f"{ep_totals['kinematic']/ep_n:>9.5f} "
              f"{ep_totals['bc']/ep_n:>9.5f} "
              f"{ep_totals['moment']/ep_n:>9.5f} "
              f"{val_mse:>9.5f}")

pinn_model.load_state_dict(best_pinn_state)
print("BeamPINN Training Complete!")

# %% [markdown]
# # Step 8: Compare Models & Evaluate Physical Consistency

# %%
def predict_beamnet(model, X_sc, scaler_y):
    model.eval()
    with torch.no_grad():
        pred_sc = model(torch.tensor(X_sc).to(device)).cpu().numpy()
    return scaler_y.inverse_transform(pred_sc)

def predict_beampinn(model, X_raw):
    model.eval()
    with torch.no_grad():
        pred_raw = model.forward_raw(torch.tensor(X_raw).to(device)).cpu().numpy()
    return pred_raw

def kinematic_residual(model, X_raw, use_pinn=True):
    Xt = torch.tensor(X_raw).to(device)
    sec = Xt[:, SECTION_IDX:SECTION_IDX+1].detach().requires_grad_(True)
    X_with_sec = torch.cat([Xt[:, :SECTION_IDX], sec, Xt[:, SECTION_IDX+1:]], dim=1)
    
    if use_pinn:
        pred = model.forward_raw(X_with_sec)
        slope_p = pred[:, 0:1]
        w_p = pred[:, 1:2]
    else:
        # Scale manually for standard network
        X_mean_t = torch.tensor(scaler_X.mean_, dtype=torch.float32, device=device)
        X_std_t = torch.tensor(scaler_X.scale_, dtype=torch.float32, device=device)
        y_mean_t = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=device)
        y_std_t = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=device)
        
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
    
    return float((slope_p - dw_dx).abs().mean().item())

# Make predictions on test set
net_preds = predict_beamnet(net_model, X_test_sc, scaler_y)
pinn_preds = predict_beampinn(pinn_model, X_test_raw)

# Calculate metrics
def compute_metrics(truth, preds):
    metrics = {}
    for i, name in enumerate(["Slope", "Deflection"]):
        mae = mean_absolute_error(truth[:, i], preds[:, i])
        r2 = r2_score(truth[:, i], preds[:, i])
        metrics[name] = {"MAE": mae, "R2": r2}
    return metrics

net_metrics = compute_metrics(y_test_raw, net_preds)
pinn_metrics = compute_metrics(y_test_raw, pinn_preds)

net_kin_res = kinematic_residual(net_model, X_test_raw, use_pinn=False)
pinn_kin_res = kinematic_residual(pinn_model, X_test_raw, use_pinn=True)

# Print Summary Table
print("\n" + "=" * 55)
print(f"  {'Metric':<18} | {'BeamNet (Standard)':<20} | {'BeamPINN (Physics)':<20}")
print("-" * 65)
print(f"  {'Slope R²':<18} | {net_metrics['Slope']['R2']:<20.4f} | {pinn_metrics['Slope']['R2']:<20.4f}")
print(f"  {'Slope MAE':<18} | {net_metrics['Slope']['MAE']:<20.6f} | {pinn_metrics['Slope']['MAE']:<20.6f}")
print(f"  {'Deflection R²':<18} | {net_metrics['Deflection']['R2']:<20.4f} | {pinn_metrics['Deflection']['R2']:<20.4f}")
print(f"  {'Deflection MAE':<18} | {net_metrics['Deflection']['MAE']:<20.6f} | {pinn_metrics['Deflection']['MAE']:<20.6f}")
print(f"  {'Kinematic Residual':<18} | {net_kin_res:<20.6f} | {pinn_kin_res:<20.6f}")
print("=" * 55)

# %% [markdown]
# # Step 9: Plot & Visualize Results

# %%
# Plot Styling (Clean Light Theme)
plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
    "axes.edgecolor": "#000000",
    "axes.grid": True,
    "grid.color": "#CCCCCC",
    "grid.alpha": 0.5,
    "font.size": 12,
})

targets = ["Slope", "Deflection"]
colors = {"BeamNet": "#FF6B6B", "BeamPINN": "#00E676"}

fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

for i, target in enumerate(targets):
    truth = y_test_raw[:, i]
    n_pred = net_preds[:, i]
    p_pred = pinn_preds[:, i]

    # 1. Scatter: Predicted vs True
    ax = axes[i, 0]
    ax.scatter(truth, n_pred, s=4, alpha=0.3, color=colors["BeamNet"],
               label=f"BeamNet (R²={net_metrics[target]['R2']:.4f})")
    ax.scatter(truth, p_pred, s=4, alpha=0.3, color=colors["BeamPINN"],
               label=f"BeamPINN (R²={pinn_metrics[target]['R2']:.4f})")
    lims = [min(truth.min(), n_pred.min(), p_pred.min()), max(truth.max(), n_pred.max(), p_pred.max())]
    ax.plot(lims, lims, "--", color="#555555", lw=1.5, alpha=0.7)
    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title(f"{target} — Predicted vs True")
    ax.legend(markerscale=4)
    ax.set_aspect("equal")

    # 2. Residuals
    ax = axes[i, 1]
    n_res = truth - n_pred
    p_res = truth - p_pred
    ax.scatter(truth, n_res, s=4, alpha=0.3, color=colors["BeamNet"], label="BeamNet")
    ax.scatter(truth, p_res, s=4, alpha=0.3, color=colors["BeamPINN"], label="BeamPINN")
    ax.axhline(0, color="#555555", lw=1.5, ls="--", alpha=0.7)
    ax.set_xlabel("True Value")
    ax.set_ylabel("Residual Error")
    ax.set_title(f"{target} — Residuals")
    ax.legend(markerscale=4)

    # 3. Error Histogram
    ax = axes[i, 2]
    ax.hist(n_res, bins=60, alpha=0.5, color=colors["BeamNet"], label=f"BeamNet (std={n_res.std():.4f})")
    ax.hist(p_res, bins=60, alpha=0.5, color=colors["BeamPINN"], label=f"BeamPINN (std={p_res.std():.4f})")
    ax.set_xlabel("Error Value")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{target} — Error Distribution")
    ax.legend()

fig.suptitle("BeamNet vs BeamPINN Comparison Dashboard", fontsize=16, fontweight="bold", y=1.02)
plt.show()
