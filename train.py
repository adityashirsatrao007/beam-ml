"""
Beam Slope & Deflection — ML Training Script
=============================================
Trains a neural network to predict slope and deflection of single-span beams
given: unit load position (X), beam length (L), section point, and end condition.

Dataset columns:
  Inputs : X, L, Section, End Condition
  Outputs: Slope, Deflection
"""

import os
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Optional: try to import torch; fall back to sklearn if unavailable ──────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[INFO] PyTorch not found — will use sklearn MLP as fallback.")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "data"
MODEL_DIR  = Path(__file__).parent / "models"
OUTPUT_DIR = Path(__file__).parent / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_FILE = DATA_DIR / "Slope and Deflection.csv"


# ════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    """Load and clean the beam dataset."""
    print(f"[1/5] Loading data from {CSV_FILE.name} …")
    df = pd.read_csv(CSV_FILE)

    # Rename columns to clean names
    df.columns = ["X", "L", "Section", "EndCondition", "_blank", "Slope", "Deflection"]
    df = df.drop(columns=["_blank"])

    # Drop rows with any NaN
    df = df.dropna()

    # Encode end conditions as integers (one-hot is better for neural nets)
    ec_map = {"SS": 0, "SF": 1, "FF": 2}
    df["EC_SS"] = (df["EndCondition"] == "SS").astype(float)
    df["EC_SF"] = (df["EndCondition"] == "SF").astype(float)
    df["EC_FF"] = (df["EndCondition"] == "FF").astype(float)

    print(f"    Rows loaded        : {len(df):,}")
    print(f"    End conditions     : {df['EndCondition'].value_counts().to_dict()}")
    print(f"    Slope  range       : [{df['Slope'].min():.4f}, {df['Slope'].max():.4f}]")
    print(f"    Deflection range   : [{df['Deflection'].min():.4f}, {df['Deflection'].max():.4f}]")
    return df


def prepare_features(df: pd.DataFrame):
    """Build feature matrix X and target matrix y."""
    feature_cols = ["X", "L", "Section", "EC_SS", "EC_SF", "EC_FF"]
    target_cols  = ["Slope", "Deflection"]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    return X, y, feature_cols, target_cols


# ════════════════════════════════════════════════════════════════════════════
# 2. NEURAL NETWORK DEFINITION (PyTorch)
# ════════════════════════════════════════════════════════════════════════════

class BeamNet(nn.Module):
    """
    A fully-connected neural network for beam slope & deflection prediction.

    Architecture:
      Input (6) → [128 → BN → ReLU → Dropout] × 3 → Output (2)

    Why this architecture?
      - 3 hidden layers can capture non-linear relationships between load
        position, boundary conditions, and structural response.
      - Batch Normalization stabilises training and speeds convergence.
      - Dropout prevents overfitting on the relatively small dataset.
    """

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


# ════════════════════════════════════════════════════════════════════════════
# 3. TRAINING LOOP (PyTorch)
# ════════════════════════════════════════════════════════════════════════════

def train_torch(X_train, X_val, y_train, y_val,
                epochs: int = 150, batch_size: int = 1024, lr: float = 1e-3):
    """Train BeamNet with Adam + ReduceLROnPlateau scheduler."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device : {device}")

    # Tensors
    Xt = torch.tensor(X_train).to(device)
    yt = torch.tensor(y_train).to(device)
    Xv = torch.tensor(X_val).to(device)
    yv = torch.tensor(y_val).to(device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    model = BeamNet().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=10, factor=0.5, min_lr=1e-6
    )
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    print(f"\n{'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}  {'LR':>10}")
    print("─" * 50)

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        epoch_loss = 0.0
        for Xb, yb in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(Xb)
        train_mse = epoch_loss / len(Xt)

        # ── validate ──
        model.eval()
        with torch.no_grad():
            val_mse = loss_fn(model(Xv), yv).item()

        scheduler.step(val_mse)
        history["train_loss"].append(train_mse)
        history["val_loss"].append(val_mse)

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimiser.param_groups[0]["lr"]
            print(f"{epoch:>6}  {train_mse:>12.6f}  {val_mse:>12.6f}  {lr_now:>10.2e}")

    # Restore best weights
    model.load_state_dict(best_state)
    return model, history, device


# ════════════════════════════════════════════════════════════════════════════
# 4. SKLEARN FALLBACK (if PyTorch not available)
# ════════════════════════════════════════════════════════════════════════════

def train_sklearn(X_train, X_val, y_train, y_val):
    """Train sklearn MLPRegressor as fallback."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.multioutput import MultiOutputRegressor

    model = MLPRegressor(
        hidden_layer_sizes=(128, 128, 64),
        activation="relu",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION & REPORTING
# ════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test, scaler_y, device=None, use_torch=True):
    """Compute MAE and R² for slope and deflection separately."""
    if use_torch and device is not None:
        model.eval()
        with torch.no_grad():
            preds_scaled = model(torch.tensor(X_test).to(device)).cpu().numpy()
    else:
        preds_scaled = model.predict(X_test)

    # Inverse-transform predictions and true values
    preds = scaler_y.inverse_transform(preds_scaled)
    truth = scaler_y.inverse_transform(y_test)

    results = {}
    for i, name in enumerate(["Slope", "Deflection"]):
        mae = mean_absolute_error(truth[:, i], preds[:, i])
        r2  = r2_score(truth[:, i], preds[:, i])
        results[name] = {"MAE": round(mae, 6), "R2": round(r2, 6)}
        print(f"    {name:12s} → MAE = {mae:.6f}  |  R² = {r2:.6f}")

    return results, preds, truth


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  BEAM SLOPE & DEFLECTION — ML TRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Load & prepare ──────────────────────────────────────────────────
    df = load_data()
    X_raw, y_raw, feature_cols, target_cols = prepare_features(df)

    # ── 2. Scale features ──────────────────────────────────────────────────
    print("\n[2/5] Scaling features …")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw).astype(np.float32)
    y_scaled = scaler_y.fit_transform(y_raw).astype(np.float32)

    # ── 3. Split ───────────────────────────────────────────────────────────
    print("[3/5] Splitting dataset (70% train / 15% val / 15% test) …")
    X_tv, X_test, y_tv, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.15 / 0.85, random_state=42
    )
    print(f"    Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

    # ── 4. Train ───────────────────────────────────────────────────────────
    print("\n[4/5] Training …")
    use_torch = TORCH_AVAILABLE

    if use_torch:
        model, history, device = train_torch(X_train, X_val, y_train, y_val,
                                              epochs=150, batch_size=512)
        # Save PyTorch model
        torch.save(model.state_dict(), MODEL_DIR / "beamnet.pt")
        with open(OUTPUT_DIR / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(f"\n    Model saved → models/beamnet.pt")
    else:
        device = None
        model = train_sklearn(X_train, X_val, y_train, y_val)
        joblib.dump(model, MODEL_DIR / "beamnet_sklearn.pkl")
        print(f"\n    Model saved → models/beamnet_sklearn.pkl")

    # Save scalers (needed for inference)
    joblib.dump(scaler_X, MODEL_DIR / "scaler_X.pkl")
    joblib.dump(scaler_y, MODEL_DIR / "scaler_y.pkl")
    print(f"    Scalers saved → models/scaler_X.pkl, scaler_y.pkl")

    # ── 5. Evaluate ────────────────────────────────────────────────────────
    print("\n[5/5] Evaluation on held-out test set …")
    results, preds, truth = evaluate(
        model, X_test, y_test, scaler_y, device=device, use_torch=use_torch
    )

    # Save summary
    summary = {
        "model": "BeamNet (PyTorch)" if use_torch else "MLPRegressor (sklearn)",
        "train_rows": len(X_train),
        "val_rows":   len(X_val),
        "test_rows":  len(X_test),
        "features":   feature_cols,
        "targets":    target_cols,
        "metrics":    results,
    }
    with open(OUTPUT_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print(f"  Slope     R² = {results['Slope']['R2']:.4f}")
    print(f"  Deflection R² = {results['Deflection']['R2']:.4f}")
    print(f"  Results saved → outputs/evaluation_summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
