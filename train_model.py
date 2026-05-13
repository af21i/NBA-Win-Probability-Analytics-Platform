from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import WinProbabilityNet


DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "period",
    "seconds_remaining",
    "home_score",
    "away_score",
    "score_diff_home",
    "abs_score_diff",
    "possessions_proxy",
    "home_fouls_proxy",
    "away_fouls_proxy",
    "is_clutch_time",

    "home_pre_wins",
    "home_pre_losses",
    "away_pre_wins",
    "away_pre_losses",
    "home_pre_win_pct",
    "away_pre_win_pct",
    "pre_game_win_pct_diff",
]


def main():
    df = pd.read_csv(DATA_DIR / "model_dataset.csv")

    X = df[FEATURES].values.astype("float32")
    y = df["home_win"].values.astype("float32").reshape(-1, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_val = scaler.transform(X_val).astype("float32")

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_x = torch.tensor(X_val)
    val_y = torch.tensor(y_val)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    model = WinProbabilityNet(input_dim=len(FEATURES))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 31):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = criterion(val_pred, val_y).item()
            val_acc = ((val_pred >= 0.5).float() == val_y).float().mean().item()

        print(
            f"Epoch {epoch:02d} | train loss {total_loss / len(train_loader):.4f} "
            f"| val loss {val_loss:.4f} | val acc {val_acc:.3f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": FEATURES,
            "input_dim": len(FEATURES),
        },
        MODEL_DIR / "winprob_model.pt",
    )

    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    print("Saved models/winprob_model.pt and models/scaler.pkl")


if __name__ == "__main__":
    main()