"""
train_lstm.py
-------------
Reshapes flow features as single-timestep sequences and trains a stacked
LSTM network.  Saved to models/lstm_model.keras.
"""

import os
import numpy as np
import tensorflow as tf
from pathlib import Path

os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)
tf.random.set_seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"

TIMESTEPS  = 1
EPOCHS     = 20
BATCH_SIZE = 64
N_CLASSES  = 8


def build_model(n_features: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True,
                             input_shape=(TIMESTEPS, n_features)),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
    ], name="lstm_nids")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    print("\n=== LSTM Training ===")

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

    # Reshape to (samples, timesteps=1, features)
    X_train_seq = X_train.reshape(-1, TIMESTEPS, X_train.shape[1])
    X_val_seq   = X_val.reshape(-1,   TIMESTEPS, X_val.shape[1])
    print(f"  Reshaped X_train: {X_train_seq.shape}   X_val: {X_val_seq.shape}")

    # One-hot encode labels
    y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=N_CLASSES)
    y_val_oh   = tf.keras.utils.to_categorical(y_val,   num_classes=N_CLASSES)

    n_features = X_train.shape[1]
    model = build_model(n_features)
    model.summary()

    print(f"\n  Training for {EPOCHS} epochs (batch={BATCH_SIZE}) …")
    history = model.fit(
        X_train_seq, y_train_oh,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val_seq, y_val_oh),
        verbose=1,
    )

    out_path = MODELS_DIR / "lstm_model.keras"
    model.save(str(out_path))
    print(f"  Model saved → {out_path}")

    np.save(MODELS_DIR / "lstm_history.npy", history.history)
    print(f"  History saved → {MODELS_DIR / 'lstm_history.npy'}")
    print("\n✓ LSTM training complete.")


if __name__ == "__main__":
    main()
