from pathlib import Path

import joblib
import pandas as pd
import torch

from src.model import WinProbabilityNet


MODEL_PATH = Path("models/winprob_model.pt")
SCALER_PATH = Path("models/scaler.pkl")


class WinProbabilityPredictor:
    def __init__(self):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        self.features = checkpoint["features"]
        self.scaler = joblib.load(SCALER_PATH)

        self.model = WinProbabilityNet(input_dim=checkpoint["input_dim"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_home_win_probability(self, feature_dict: dict) -> float:
        row = pd.DataFrame([{k: feature_dict[k] for k in self.features}])
        x = self.scaler.transform(row.values.astype("float32")).astype("float32")

        with torch.no_grad():
            prob = self.model(torch.tensor(x)).item()

        return float(prob)
