"""
preprocess.py
-------------
Loads all CICIDS2017 CSV files, cleans and normalises them,
encodes labels, performs a stratified train/val/test split,
applies SMOTE to the training set, and saves all artefacts.
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Label mapping ──────────────────────────────────────────────────────────────
LABEL_MAP = {
    # Benign
    "benign":                       "BENIGN",
    # DoS
    "dos hulk":                     "DoS",
    "dos goldeneye":                "DoS",
    "dos slowloris":                "DoS",
    "dos slowhttptest":             "DoS",
    "heartbleed":                   "DoS",
    # DDoS
    "ddos":                         "DDoS",
    # PortScan
    "portscan":                     "PortScan",
    # Brute Force
    "ftp-patator":                  "Brute Force",
    "ssh-patator":                  "Brute Force",
    # Botnet
    "bot":                          "Botnet",
    # Infiltration
    "infiltration":                 "Infiltration",
    # Web Attack
    "web attack \x96 brute force":  "Web Attack",
    "web attack \x96 xss":          "Web Attack",
    "web attack \x96 sql injection":"Web Attack",
    "web attack – brute force":     "Web Attack",
    "web attack – xss":             "Web Attack",
    "web attack – sql injection":   "Web Attack",
    "web attack - brute force":     "Web Attack",
    "web attack - xss":             "Web Attack",
    "web attack - sql injection":   "Web Attack",
}


def load_and_concatenate(data_raw: Path) -> pd.DataFrame:
    """Load every CSV in data/raw and concatenate into one DataFrame."""
    csv_files = sorted(glob.glob(str(data_raw / "*.csv")))
    if not csv_files:
        sys.exit(f"[ERROR] No CSV files found in {data_raw}")

    frames = []
    for i, fp in enumerate(csv_files, 1):
        print(f"  [{i}/{len(csv_files)}] Loading {os.path.basename(fp)} …")
        df = pd.read_csv(fp, low_memory=False, encoding="utf-8", encoding_errors="replace")
        frames.append(df)

    print(f"  Concatenating {len(frames)} files …")
    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total rows before cleaning: {len(combined):,}")
    return combined


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalise column names to lowercase + underscores."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove NaN, inf, and -inf rows."""
    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f"  Removed {n_before - len(df):,} invalid rows  →  {len(df):,} rows remain")
    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise the label column using LABEL_MAP."""
    # Find the label column (could be 'label' or ' Label' etc.)
    label_col = None
    for col in df.columns:
        if col.strip().lower() in ("label", " label"):
            label_col = col
            break
        if "label" in col:
            label_col = col
            break

    if label_col is None:
        sys.exit("[ERROR] Could not find a 'label' column in the data.")

    if label_col != "label":
        df = df.rename(columns={label_col: "label"})

    df["label"] = df["label"].str.strip().str.lower().map(
        lambda x: LABEL_MAP.get(x, x)
    )
    # Drop rows with unmapped labels (keep only the 8 known classes)
    known = set(LABEL_MAP.values())
    before = len(df)
    df = df[df["label"].isin(known)]
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped:,} rows with unrecognised labels.")

    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


def split_features_labels(df: pd.DataFrame):
    """Return X (feature matrix) and y (label series)."""
    X = df.drop(columns=["label"]).select_dtypes(include=[np.number])
    y = df["label"]
    return X, y


def main():
    print("\n=== Step 1 — Loading CSV files ===")
    df = load_and_concatenate(DATA_RAW)

    print("\n=== Step 2 — Standardising column names ===")
    df = standardise_columns(df)

    print("\n=== Step 3 — Cleaning (NaN / Inf removal) ===")
    df = clean(df)

    print("\n=== Step 4 — Mapping labels ===")
    df = map_labels(df)

    print("\n=== Step 5 — Splitting features and labels ===")
    X, y = split_features_labels(df)
    print(f"  Features: {X.shape[1]}   Samples: {X.shape[0]:,}")

    print("\n=== Step 6 — Fitting MinMaxScaler ===")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    print(f"  Scaler saved to {MODELS_DIR / 'scaler.joblib'}")

    print("\n=== Step 7 — Encoding labels ===")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    print(f"  Classes: {list(le.classes_)}")
    print(f"  Encoder saved to {MODELS_DIR / 'label_encoder.joblib'}")

    print("\n=== Step 8 — Stratified train / val / test split (70/15/15) ===")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_encoded,
        test_size=0.30,
        random_state=42,
        stratify=y_encoded,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )
    print(f"  Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")

    print("\n=== Step 9 — Applying SMOTE to training set only ===")
    print("  This may take several minutes on the full dataset …")

    # Cast to float32: halves memory compared to float64
    X_train = X_train.astype(np.float32)

    counts        = Counter(y_train.tolist())
    majority_count = max(counts.values())
    # Cap target: bring each minority class up to at most 10 % of the majority,
    # or 10× its current size — whichever is SMALLER.  This prevents SMOTE from
    # trying to synthesise millions of rows and running out of memory.
    smote_cap = max(majority_count // 10, 5_000)

    K_NEIGHBORS = 5   # default SMOTE k

    # Separate classes too small for SMOTE (need at least k+1 samples)
    smote_strategy = {}
    ros_strategy   = {}
    for cls, cnt in counts.items():
        if cnt >= majority_count:          # already the majority — skip
            continue
        target = min(cnt * 10, smote_cap)
        target = max(target, cnt)          # never shrink
        if cnt >= K_NEIGHBORS + 1:
            smote_strategy[cls] = target
        else:
            ros_strategy[cls] = target

    print(f"  SMOTE targets  : {smote_strategy}")
    print(f"  ROS targets    : {ros_strategy}")

    # Step A: RandomOverSampler for tiny classes first
    if ros_strategy:
        # Include all classes in ROS strategy (keep majority & SMOTE classes as-is)
        ros_full = {cls: cnt for cls, cnt in counts.items()}
        ros_full.update(ros_strategy)
        ros = RandomOverSampler(sampling_strategy=ros_strategy, random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        print(f"  After ROS: {X_train.shape[0]:,} rows")
        # Recalculate counts after ROS
        counts = Counter(y_train.tolist())

    # Step B: SMOTE for the rest of the minority classes
    if smote_strategy:
        smote = SMOTE(
            sampling_strategy=smote_strategy,
            random_state=42,
            k_neighbors=K_NEIGHBORS,
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"  Training set after oversampling: {X_train.shape[0]:,} rows")
    print(f"  Class distribution: {dict(sorted(Counter(y_train.tolist()).items()))}")

    X_train_res, y_train_res = X_train, y_train


    print("\n=== Step 10 — Saving arrays ===")
    np.save(DATA_DIR / "X_train.npy", X_train_res.astype(np.float32))
    np.save(DATA_DIR / "y_train.npy", y_train_res)
    np.save(DATA_DIR / "X_val.npy",   X_val.astype(np.float32))
    np.save(DATA_DIR / "y_val.npy",   y_val)
    np.save(DATA_DIR / "X_test.npy",  X_test.astype(np.float32))
    np.save(DATA_DIR / "y_test.npy",  y_test)
    # Also save feature column names (before feature selection)
    np.save(DATA_DIR / "feature_names.npy", np.array(X.columns.tolist()))
    print(f"  Saved X_train, y_train, X_val, y_val, X_test, y_test → {DATA_DIR}")
    print("\n✓ Preprocessing complete.")


if __name__ == "__main__":
    main()
