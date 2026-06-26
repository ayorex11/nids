"""
train_dnn.py
------------
Builds and trains a feedforward Deep Neural Network using TensorFlow / Keras.
Saves the trained model to models/dnn_model.keras and the training history
to models/dnn_history.npy.
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
tf.random.set_seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"

EPOCHS     = 20
BATCH_SIZE = 64
N_CLASSES  = 8


def build_model(n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
    ], name="dnn_nids")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("\n=== DNN Training ===")

    print("  Loading data …")
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val   = np.load(DATA_DIR / "X_val.npy")
    y_val   = np.load(DATA_DIR / "y_val.npy")
    print(f"  X_train: {X_train.shape}   X_val: {X_val.shape}")

    # Stratified 150k subset for CPU-feasible training
    from sklearn.utils import resample
    X_train, y_train = resample(
        X_train, y_train,
        n_samples=150000,
        stratify=y_train,
        random_state=42
    )
    print(f"  Training on {X_train.shape[0]} samples")

    # One-hot encode labels
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=N_CLASSES)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes=N_CLASSES)

    n_features = X_train.shape[1]
    model = build_model(n_features)
    model.summary()

    print(f"\n  Training for {EPOCHS} epochs (batch={BATCH_SIZE}) …")
    history = model.fit(
        X_train, y_train_oh,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val_oh),
        verbose=1,
    )

    out_path = MODELS_DIR / "dnn_model.keras"
    model.save(str(out_path))
    print(f"  Model saved → {out_path}")

    # Save history for plotting
    np.save(MODELS_DIR / "dnn_history.npy", history.history)
    print(f"  History saved → {MODELS_DIR / 'dnn_history.npy'}")
    print("\n✓ DNN training complete.")


if __name__ == "__main__":
    main()
