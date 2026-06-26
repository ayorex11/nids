"""
feature_selection.py
--------------------
Step 1 – Drop highly correlated features (|r| > 0.95).
Step 2 – Train a preliminary Random Forest and keep top-25 features by importance.
Overwrites X_train, X_val, X_test with the reduced feature set.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"

TOP_N        = 25
CORR_THRESH  = 0.95


def drop_correlated(X: np.ndarray, threshold: float = CORR_THRESH):
    """Return column indices to keep after removing highly correlated features."""
    print(f"  Computing correlation matrix ({X.shape[1]} features) …")
    corr = np.corrcoef(X, rowvar=False)
    corr = np.abs(corr)

    to_drop = set()
    n = corr.shape[0]
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            if corr[i, j] > threshold:
                to_drop.add(j)

    keep = [i for i in range(n) if i not in to_drop]
    print(f"  Dropped {len(to_drop)} correlated features  →  {len(keep)} remain")
    return keep


def top_importance(X: np.ndarray, y: np.ndarray, keep_idx: list, top_n: int = TOP_N):
    """Train a quick RF and return the indices (in original space) of the top-N features."""
    print(f"  Training preliminary Random Forest on {X.shape[1]} features …")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = rf.feature_importances_
    ranked = np.argsort(importances)[::-1][:top_n]
    # Map back to original feature indices
    selected_original = [keep_idx[r] for r in ranked]
    print(f"  Selected top-{top_n} features by importance.")
    return sorted(selected_original)   # keep sorted for reproducibility


def main():
    print("\n=== Feature Selection ===")

    print("  Loading training data …")
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val   = np.load(DATA_DIR / "X_val.npy")
    X_test  = np.load(DATA_DIR / "X_test.npy")
    print(f"  X_train: {X_train.shape}   X_val: {X_val.shape}   X_test: {X_test.shape}")

    # ── Step 1: correlation filtering ─────────────────────────────────────────
    print("\n--- Step 1: Correlation filtering ---")
    keep_idx = drop_correlated(X_train)

    X_train_filtered = X_train[:, keep_idx]
    X_val_filtered   = X_val[:,   keep_idx]
    X_test_filtered  = X_test[:,  keep_idx]

    # ── Step 2: feature importance ────────────────────────────────────────────
    print("\n--- Step 2: Feature importance (Random Forest) ---")
    selected_idx_local = top_importance(X_train_filtered, y_train,
                                        keep_idx=list(range(len(keep_idx))))

    # Express selected indices in the *original* feature space
    selected_original = [keep_idx[i] for i in selected_idx_local]
    selected_original_sorted = sorted(selected_original)

    # ── Save selected feature indices ─────────────────────────────────────────
    np.save(MODELS_DIR / "selected_features.npy",
            np.array(selected_original_sorted, dtype=np.int64))
    print(f"  Saved selected_features.npy  →  {selected_original_sorted}")

    # ── Apply mask and overwrite .npy files ───────────────────────────────────
    X_train_sel = X_train[:, selected_original_sorted]
    X_val_sel   = X_val[:,   selected_original_sorted]
    X_test_sel  = X_test[:,  selected_original_sorted]

    np.save(DATA_DIR / "X_train.npy", X_train_sel)
    np.save(DATA_DIR / "X_val.npy",   X_val_sel)
    np.save(DATA_DIR / "X_test.npy",  X_test_sel)

    print(f"\n  Final shapes:")
    print(f"    X_train: {X_train_sel.shape}")
    print(f"    X_val:   {X_val_sel.shape}")
    print(f"    X_test:  {X_test_sel.shape}")
    print("\n✓ Feature selection complete.")


if __name__ == "__main__":
    main()
