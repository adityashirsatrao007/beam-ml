import os
import sys
import numpy as np
import joblib
import torch
from flask import Flask, request, jsonify, render_template

# Ensure path has the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import BeamNet
from train_pinn import BeamPINN

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Load scalers
scaler_X = joblib.load(os.path.join(MODEL_DIR, "scaler_X.pkl"))
scaler_y = joblib.load(os.path.join(MODEL_DIR, "scaler_y.pkl"))

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. BeamNet
net_model = BeamNet()
net_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "beamnet.pt"), map_location=device))
net_model.to(device)
net_model.eval()

# 2. BeamPINN
pinn_model = BeamPINN(
    X_mean=scaler_X.mean_, X_std=scaler_X.scale_,
    y_mean=scaler_y.mean_, y_std=scaler_y.scale_
)
pinn_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "beampinn.pt"), map_location=device)["state_dict"])
pinn_model.to(device)
pinn_model.eval()

EC_MAP = {"SS": [1, 0, 0], "SF": [0, 1, 0], "FF": [0, 0, 1]}

def predict_model(model, X, L, Section, end_condition, is_pinn=False):
    ec = EC_MAP[end_condition]
    raw = np.array([[X, L, Section, *ec]], dtype=np.float32)
    
    if is_pinn:
        with torch.no_grad():
            Xt = torch.tensor(raw).to(device)
            pred_raw = model.forward_raw(Xt).cpu().numpy()[0]
        return {"Slope": float(pred_raw[0]), "Deflection": float(pred_raw[1])}
    else:
        scaled = scaler_X.transform(raw).astype(np.float32)
        with torch.no_grad():
            Xt = torch.tensor(scaled).to(device)
            pred_scaled = model(Xt).cpu().numpy()
        pred = scaler_y.inverse_transform(pred_scaled)[0]
        return {"Slope": float(pred[0]), "Deflection": float(pred[1])}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["GET"])
def predict():
    try:
        X = float(request.args.get("X", 2.5))
        L = float(request.args.get("L", 5.0))
        Section = float(request.args.get("Section", 2.5))
        end_condition = request.args.get("EndCondition", "SS")
        
        if end_condition not in EC_MAP:
            return jsonify({"error": f"Invalid EndCondition. Must be one of {list(EC_MAP.keys())}"}), 400
            
        net_pred = predict_model(net_model, X, L, Section, end_condition, is_pinn=False)
        pinn_pred = predict_model(pinn_model, X, L, Section, end_condition, is_pinn=True)
        
        return jsonify({
            "BeamNet": net_pred,
            "BeamPINN": pinn_pred
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/curve", methods=["GET"])
def curve():
    try:
        X = float(request.args.get("X", 2.5))
        L = float(request.args.get("L", 5.0))
        end_condition = request.args.get("EndCondition", "SS")
        
        if end_condition not in EC_MAP:
            return jsonify({"error": "Invalid EndCondition"}), 400
            
        sections = np.linspace(0, L, 50, dtype=np.float32)
        net_curve = []
        pinn_curve = []
        
        # Build batch request for efficiency
        ec = EC_MAP[end_condition]
        raw_batch = np.array([[X, L, s, *ec] for s in sections], dtype=np.float32)
        
        # BeamNet predictions
        scaled_batch = scaler_X.transform(raw_batch).astype(np.float32)
        with torch.no_grad():
            Xt_net = torch.tensor(scaled_batch).to(device)
            pred_net_scaled = net_model(Xt_net).cpu().numpy()
            pred_net = scaler_y.inverse_transform(pred_net_scaled)
            
        # BeamPINN predictions
        with torch.no_grad():
            Xt_pinn = torch.tensor(raw_batch).to(device)
            pred_pinn = pinn_model.forward_raw(Xt_pinn).cpu().numpy()
            
        result = {
            "sections": [float(s) for s in sections],
            "BeamNet": {
                "Slope": [float(y[0]) for y in pred_net],
                "Deflection": [float(y[1]) for y in pred_net]
            },
            "BeamPINN": {
                "Slope": [float(y[0]) for y in pred_pinn],
                "Deflection": [float(y[1]) for y in pred_pinn]
            }
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
