"""
train_rf.py
-----------
Trains a Random Forest classifier on the pre-processed, feature-selected
CICIDS2017 data and saves the model to models/random_forest.joblib.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"


def main():
    print("\n=== Random Forest Training ===")

    print("  Loading data …")
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val   = np.load(DATA_DIR / "X_val.npy")
    y_val   = np.load(DATA_DIR / "y_val.npy")
    print(f"  X_train: {X_train.shape}   X_val: {X_val.shape}")

    print("  Training Random Forest (100 estimators, gini, n_jobs=-1) …")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        criterion="gini",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, rf.predict(X_train))
    val_acc   = accuracy_score(y_val,   rf.predict(X_val))
    print(f"  Training accuracy : {train_acc:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")

    out_path = MODELS_DIR / "random_forest.joblib"
    joblib.dump(rf, out_path)
    print(f"  Model saved → {out_path}")
    print("\n✓ Random Forest training complete.")


if __name__ == "__main__":
    main()
