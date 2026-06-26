"""
evaluate.py
-----------
Loads all four trained models, runs inference on the test set,
and produces per-class metrics, plots, and a comparison CSV.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

np.random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_CLASSES = 8


# ── Helpers ────────────────────────────────────────────────────────────────────
def false_positive_rate(y_true_bin, y_score):
    """Macro-averaged false positive rate (binary per class then average)."""
    fprs = []
    for c in range(y_true_bin.shape[1]):
        tn = np.sum((y_true_bin[:, c] == 0) & (y_score[:, c] == 0))
        fp = np.sum((y_true_bin[:, c] == 0) & (y_score[:, c] == 1))
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    return fprs


def per_class_fpr(y_true, y_pred, n_classes):
    fprs = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        tn = np.sum((y_pred != c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fprs.append(fpr)
    return fprs


def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    out = RESULTS_DIR / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    Saved → {out}")


def plot_roc(y_true_bin, y_prob, class_names, model_name):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for i, (cls, col) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=1.5,
                label=f"{cls} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (one-vs-rest) — {model_name}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out = RESULTS_DIR / f"roc_{model_name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    Saved → {out}")


def plot_training_curves(history_path, model_name):
    if not history_path.exists():
        print(f"    [SKIP] No history found for {model_name}")
        return
    history = np.load(history_path, allow_pickle=True).item()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["loss"],     label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].legend()
    axes[1].plot(history["accuracy"],     label="Train Acc")
    axes[1].plot(history["val_accuracy"], label="Val Acc")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].legend()
    plt.tight_layout()
    out = RESULTS_DIR / f"training_curve_{model_name.replace(' ', '_')}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"    Saved → {out}")


# ── Inline NIDSEnv (avoids import path dependency) ────────────────────────────
class _NIDSEnv(gym.Env):
    """Minimal NIDS environment for inference-only use in evaluation."""
    metadata = {"render_modes": []}

    def __init__(self, X, y, benign_class=0):
        super().__init__()
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.n_samples   = len(X)
        self.n_classes   = int(y.max()) + 1
        self.benign_class = benign_class
        self.current_idx  = 0
        n_features = X.shape[1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_classes)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        return self.X[0], {}

    def step(self, action):
        self.current_idx += 1
        terminated = (self.current_idx >= self.n_samples)
        obs = self.X[0] if terminated else self.X[self.current_idx]
        return obs, 0.0, terminated, False, {}

    def render(self):
        pass


# ── DQN inference ──────────────────────────────────────────────────────────────
def dqn_predict(model, X_test):
    """Step through X_test using the trained DQN policy."""
    import torch

    preds  = []
    q_vals = []

    for i in range(len(X_test)):
        obs = X_test[i].astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = model.q_net(obs_t).cpu().numpy()[0]
        preds.append(int(action))
        q_vals.append(q)

    return np.array(preds), np.array(q_vals)


# ── Softmax proba from Q-values (for ROC) ─────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Evaluation ===")

    # ── Load test data & encoder ───────────────────────────────────────────────
    print("  Loading test data …")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    le     = joblib.load(MODELS_DIR / "label_encoder.joblib")
    class_names = list(le.classes_)
    print(f"  Classes: {class_names}")

    y_test_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))

    # ── Load models ────────────────────────────────────────────────────────────
    print("  Loading models …")
    import tensorflow as tf

    rf_model   = joblib.load(MODELS_DIR / "random_forest.joblib")
    dnn_model  = tf.keras.models.load_model(str(MODELS_DIR / "dnn_model.keras"))
    lstm_model = tf.keras.models.load_model(str(MODELS_DIR / "lstm_model.keras"))

    from stable_baselines3 import DQN as SB3DQN
    dqn_model = SB3DQN.load(str(MODELS_DIR / "dqn_agent"))

    # ── Inference ──────────────────────────────────────────────────────────────
    print("  Running inference …")

    # RF
    rf_pred  = rf_model.predict(X_test)
    rf_prob  = rf_model.predict_proba(X_test)

    # DNN
    dnn_prob = dnn_model.predict(X_test, verbose=0)
    dnn_pred = np.argmax(dnn_prob, axis=1)

    # LSTM
    X_test_seq  = X_test.reshape(-1, 1, X_test.shape[1])
    lstm_prob   = lstm_model.predict(X_test_seq, verbose=0)
    lstm_pred   = np.argmax(lstm_prob, axis=1)

    # DQN
    print("  Running DQN inference (stepping through test set) …")
    dqn_pred, dqn_q = dqn_predict(dqn_model, X_test.astype(np.float32))
    dqn_prob = softmax(dqn_q)

    models_info = [
        ("Random Forest",      rf_pred,   rf_prob),
        ("Deep Neural Network",dnn_pred,  dnn_prob),
        ("LSTM",               lstm_pred, lstm_prob),
        ("DQN Agent",          dqn_pred,  dqn_prob),
    ]

    # ── Per-model metrics ──────────────────────────────────────────────────────
    summary_rows = []

    for model_name, y_pred, y_prob in models_info:
        print(f"\n--- {model_name} ---")

        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred, average=None, zero_division=0)
        rec   = recall_score(y_test,    y_pred, average=None, zero_division=0)
        f1    = f1_score(y_test,        y_pred, average=None, zero_division=0)
        fprs  = per_class_fpr(y_test,   y_pred, N_CLASSES)

        # ROC-AUC (one-vs-rest)
        try:
            roc_auc_macro = roc_auc_score(y_test_bin, y_prob,
                                          multi_class="ovr", average="macro")
        except Exception:
            roc_auc_macro = float("nan")

        print(f"  Accuracy          : {acc:.4f}")
        print(f"  Macro Precision   : {np.mean(prec):.4f}")
        print(f"  Macro Recall      : {np.mean(rec):.4f}")
        print(f"  Macro F1          : {np.mean(f1):.4f}")
        print(f"  Macro FPR         : {np.mean(fprs):.4f}")
        print(f"  ROC-AUC (macro)   : {roc_auc_macro:.4f}")

        print(f"\n  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>8} {'FPR':>8}")
        print(f"  {'-'*58}")
        for i, cls in enumerate(class_names):
            print(f"  {cls:<20} {prec[i]:>10.4f} {rec[i]:>10.4f} "
                  f"{f1[i]:>8.4f} {fprs[i]:>8.4f}")

        # Plots
        plot_confusion_matrix(y_test, y_pred, class_names, model_name)
        plot_roc(y_test_bin, y_prob, class_names, model_name)

        summary_rows.append({
            "Model": model_name,
            "Accuracy": round(acc, 4),
            **{f"Precision_{c}": round(prec[i], 4) for i, c in enumerate(class_names)},
            **{f"Recall_{c}":    round(rec[i],  4) for i, c in enumerate(class_names)},
            **{f"F1_{c}":        round(f1[i],   4) for i, c in enumerate(class_names)},
            **{f"FPR_{c}":       round(fprs[i], 4) for i, c in enumerate(class_names)},
            "Macro_Precision": round(float(np.mean(prec)), 4),
            "Macro_Recall":    round(float(np.mean(rec)),  4),
            "Macro_F1":        round(float(np.mean(f1)),   4),
            "Macro_FPR":       round(float(np.mean(fprs)), 4),
            "ROC_AUC_macro":   round(roc_auc_macro, 4),
        })

    # ── Training curves ────────────────────────────────────────────────────────
    print("\n  Plotting training curves …")
    plot_training_curves(MODELS_DIR / "dnn_history.npy",  "Deep Neural Network")
    plot_training_curves(MODELS_DIR / "lstm_history.npy", "LSTM")

    # ── Comparison plots ───────────────────────────────────────────────────────
    print("  Plotting F1 and FPR comparison charts …")
    df_summary = pd.DataFrame(summary_rows)

    # F1 bar chart per class
    f1_cols = [f"F1_{c}" for c in class_names]
    f1_data = df_summary[["Model"] + f1_cols].set_index("Model")
    f1_data.columns = class_names

    fig, ax = plt.subplots(figsize=(14, 6))
    f1_data.T.plot(kind="bar", ax=ax, width=0.7)
    ax.set_title("F1-Score per Class — All Models")
    ax.set_xlabel("Traffic Class")
    ax.set_ylabel("F1-Score")
    ax.legend(loc="upper right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "f1_comparison.png", dpi=150)
    plt.close()
    print(f"    Saved → {RESULTS_DIR / 'f1_comparison.png'}")

    # FPR bar chart per class
    fpr_cols = [f"FPR_{c}" for c in class_names]
    fpr_data = df_summary[["Model"] + fpr_cols].set_index("Model")
    fpr_data.columns = class_names

    fig, ax = plt.subplots(figsize=(14, 6))
    fpr_data.T.plot(kind="bar", ax=ax, width=0.7)
    ax.set_title("False Positive Rate per Class — All Models")
    ax.set_xlabel("Traffic Class")
    ax.set_ylabel("FPR")
    ax.legend(loc="upper right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fpr_comparison.png", dpi=150)
    plt.close()
    print(f"    Saved → {RESULTS_DIR / 'fpr_comparison.png'}")

    # ── Save comparison CSV ────────────────────────────────────────────────────
    csv_path = RESULTS_DIR / "comparison_table.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"\n  Comparison table saved → {csv_path}")
    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()
