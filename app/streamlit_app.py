"""
streamlit_app.py
----------------
NIDS Streamlit prototype — classifies uploaded CICFlowMeter CSV files
using one of four trained models (RF, DNN, LSTM, DQN).
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIDS — Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
.main,
section.main,
div.block-container {
    background-color: #0d1117 !important;
}

#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }

html, body {
    background-color: #0d1117 !important;
    color: #ffffff !important;
}

div.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
}

button[kind="primary"],
button[kind="primary"]:hover,
button[kind="primary"]:focus {
    background-color: #00d4ff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
}

button[kind="secondary"],
button[kind="secondary"]:hover {
    background-color: #161b22 !important;
    color: #ffffff !important;
    border: 1px solid #00d4ff !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}

[data-testid="stDownloadButton"] > button {
    background-color: #00d4ff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.25rem !important;
}

[data-testid="stFileUploader"] {
    background-color: #161b22 !important;
    border-radius: 8px !important;
    border: 1px dashed #30363d !important;
}

[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
}

.stSpinner > div {
    border-color: #00d4ff transparent transparent transparent;
}

[data-testid="stAlert"] {
    background-color: #161b22 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model registry ─────────────────────────────────────────────────────────────
MODEL_NAMES = ["Random Forest", "Deep Neural Network", "LSTM", "DQN Agent"]

# ── Session state defaults ─────────────────────────────────────────────────────
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Random Forest"


# ── Load artefacts (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artefacts():
    artefacts = {}
    try:
        artefacts["scaler"]   = joblib.load(MODELS_DIR / "scaler.joblib")
        artefacts["encoder"]  = joblib.load(MODELS_DIR / "label_encoder.joblib")
        artefacts["features"] = np.load(MODELS_DIR / "selected_features.npy")
    except FileNotFoundError as e:
        artefacts["load_error"] = str(e)
        return artefacts

    errors = []

    try:
        artefacts["Random Forest"] = joblib.load(MODELS_DIR / "random_forest.joblib")
    except Exception as exc:
        errors.append(f"Random Forest: {exc}")

    try:
        import tensorflow as tf
        artefacts["Deep Neural Network"] = tf.keras.models.load_model(
            str(MODELS_DIR / "dnn_model.keras"))
    except Exception as exc:
        errors.append(f"DNN: {exc}")

    try:
        import tensorflow as tf
        artefacts["LSTM"] = tf.keras.models.load_model(
            str(MODELS_DIR / "lstm_model.keras"))
    except Exception as exc:
        errors.append(f"LSTM: {exc}")

    try:
        from stable_baselines3 import DQN as SB3DQN
        artefacts["DQN Agent"] = SB3DQN.load(str(MODELS_DIR / "dqn_agent"))
    except Exception as exc:
        errors.append(f"DQN: {exc}")

    if errors:
        artefacts["model_warnings"] = errors

    return artefacts


# ── DQN inference ──────────────────────────────────────────────────────────────
def dqn_infer(model, X: np.ndarray):
    import torch
    preds, q_maxes, q_all = [], [], []
    for i in range(len(X)):
        obs = X[i].astype(np.float32)
        action, _ = model.predict(obs, deterministic=True)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = model.q_net(obs_t).cpu().numpy()[0]
        preds.append(int(action))
        q_maxes.append(float(q.max()))
        q_all.append(q)
    q_arr = np.array(q_all)
    e = np.exp(q_arr - q_arr.max(axis=1, keepdims=True))
    proba = e / e.sum(axis=1, keepdims=True)
    return np.array(preds), np.array(q_maxes), proba


# ── Inference dispatcher ───────────────────────────────────────────────────────
def run_inference(model_name, artefacts, X):
    model   = artefacts[model_name]
    encoder = artefacts["encoder"]

    if model_name == "Random Forest":
        pred_int   = model.predict(X)
        proba      = model.predict_proba(X)
        confidence = proba.max(axis=1)

    elif model_name == "Deep Neural Network":
        proba      = model.predict(X, verbose=0)
        pred_int   = np.argmax(proba, axis=1)
        confidence = proba.max(axis=1)

    elif model_name == "LSTM":
        X_seq      = X.reshape(-1, 1, X.shape[1])
        proba      = model.predict(X_seq, verbose=0)
        pred_int   = np.argmax(proba, axis=1)
        confidence = proba.max(axis=1)

    elif model_name == "DQN Agent":
        pred_int, confidence, proba = dqn_infer(model, X)

    pred_labels = encoder.inverse_transform(pred_int)
    return pred_labels, confidence.tolist(), proba


# ── Row highlight ──────────────────────────────────────────────────────────────
def highlight_attacks(row):
    if row["Predicted Class"] != "BENIGN":
        return ["background-color: #3d1a1a"] * len(row)
    return [""] * len(row)


# ── CSV preprocessing ──────────────────────────────────────────────────────────
def prepare_upload(df_raw, artefacts):
    """
    Cleans and scales an uploaded CSV so it matches what the models expect.
    Returns (X_selected, error_message).
    """
    scaler   = artefacts["scaler"]
    feat_idx = artefacts["features"]

    # Drop label column if present
    for col in list(df_raw.columns):
        if col.strip().lower() in ("label", "class"):
            df_raw = df_raw.drop(columns=[col])
            break

    # Standardise column names exactly as preprocess.py did
    df_raw.columns = (
        df_raw.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )

    df_numeric = df_raw.select_dtypes(include=[np.number])

    if df_numeric.shape[1] == 0:
        return None, "No numeric columns found in the uploaded file."

    n_scaler_features = scaler.n_features_in_

    if df_numeric.shape[1] < n_scaler_features:
        return None, (
            f"Column mismatch: file has {df_numeric.shape[1]} numeric columns "
            f"but the scaler expects {n_scaler_features}. "
            "Ensure the CSV was produced by CICFlowMeter with the full CICIDS2017 feature set."
        )

    X_raw = df_numeric.values[:, :n_scaler_features].astype(np.float32)

    # Replace inf and NaN so the scaler does not error
    X_raw = np.where(np.isinf(X_raw), 0.0, X_raw)
    X_raw = np.where(np.isnan(X_raw), 0.0, X_raw)

    try:
        X_scaled = scaler.transform(X_raw)
    except Exception as exc:
        return None, f"Scaling error: {exc}"

    if int(feat_idx.max()) >= X_scaled.shape[1]:
        return None, (
            "Feature index out of range after scaling. "
            "Re-run feature_selection.py and retrain all models."
        )

    X_selected = X_scaled[:, feat_idx]
    return X_selected, None


# ── Error banner helper ────────────────────────────────────────────────────────
def error_banner(msg):
    st.markdown(
        f"<div style='background:#3d1a1a; border:1px solid #ff4b4b; "
        f"border-radius:8px; padding:1rem; color:#ff4b4b; margin:0.5rem 0;'>"
        f"⚠️ {msg}</div>",
        unsafe_allow_html=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():

    # Header
    st.markdown(
        "<h1 style='color:#ffffff; font-size:2.2rem; font-weight:700; "
        "margin-bottom:0.15rem; margin-top:0;'>"
        "🛡️ NIDS — Network Intrusion Detection System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#00d4ff; font-size:0.95rem; margin-top:0; margin-bottom:0.8rem;'>"
        "AI-powered traffic classification using machine learning and reinforcement learning"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<hr style='border:none; border-top:1px solid #00d4ff; margin-bottom:1.5rem;'>",
        unsafe_allow_html=True,
    )

    # Load artefacts
    with st.spinner("Loading models …"):
        artefacts = load_artefacts()

    if "load_error" in artefacts:
        error_banner(f"Failed to load artefacts: {artefacts['load_error']}<br>"
                     "Run preprocessing and training scripts first.")
        render_footer()
        return

    if "model_warnings" in artefacts:
        for w in artefacts["model_warnings"]:
            st.warning(f"Model load warning: {w}")

    # Model selector
    st.markdown(
        "<p style='font-size:0.85rem; font-weight:600; color:#ffffff; "
        "text-transform:uppercase; letter-spacing:1px; margin-bottom:0.6rem;'>"
        "Select Model</p>",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    for col, name in zip(cols, MODEL_NAMES):
        with col:
            is_active = (st.session_state.selected_model == name)
            label     = f"✓ {name}" if is_active else name
            btn_type  = "primary" if is_active else "secondary"
            if st.button(label, key=f"btn_{name}",
                         use_container_width=True, type=btn_type):
                st.session_state.selected_model = name
                st.rerun()

    active_model    = st.session_state.selected_model
    model_available = active_model in artefacts

    if not model_available:
        error_banner(f"Model <b>{active_model}</b> is not loaded. Train it first.")

    # Upload section
    st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.85rem; font-weight:600; color:#ffffff; "
        "text-transform:uppercase; letter-spacing:1px; margin-bottom:0.6rem;'>"
        "Upload Traffic CSV</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Choose a CICFlowMeter CSV file",
        type=["csv"],
        label_visibility="collapsed",
    )
    st.markdown(
        "<p style='font-size:0.8rem; color:#8b949e; margin-top:0.35rem;'>"
        "Upload a CSV file with pre-computed CICFlowMeter flow features. "
        "Column names must match the CICIDS2017 feature set.</p>",
        unsafe_allow_html=True,
    )

    # Results
    if uploaded is not None and model_available:
        st.markdown(
            "<hr style='border:none; border-top:1px solid #21262d; margin:1.2rem 0;'>",
            unsafe_allow_html=True,
        )

        try:
            df_raw = pd.read_csv(uploaded, low_memory=False,
                                 encoding="utf-8", encoding_errors="replace")
        except Exception as exc:
            error_banner(f"Could not read CSV: {exc}")
            render_footer()
            return

        X_selected, err = prepare_upload(df_raw, artefacts)

        if err:
            error_banner(err)
            render_footer()
            return

        with st.spinner(f"Running {active_model} inference …"):
            try:
                pred_labels, confidences, _ = run_inference(
                    active_model, artefacts, X_selected)
            except Exception as exc:
                error_banner(f"Inference error: {exc}")
                render_footer()
                return

        total_flows  = len(pred_labels)
        attack_flows = int(np.sum(np.array(pred_labels) != "BENIGN"))
        benign_flows = total_flows - attack_flows

        # Metric cards
        st.markdown(f"""
        <div style="display:flex; gap:16px; margin-bottom:1.5rem;">
          <div style="flex:1; background:#161b22; border:1px solid #30363d;
                      border-radius:10px; padding:1.2rem 1.5rem; text-align:center;">
            <div style="font-size:2rem; font-weight:700; color:#ffffff;
                        line-height:1; margin-bottom:0.4rem;">{total_flows:,}</div>
            <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase;
                        letter-spacing:0.8px;">Total Flows Analysed</div>
          </div>
          <div style="flex:1; background:#161b22; border:1px solid #30363d;
                      border-radius:10px; padding:1.2rem 1.5rem; text-align:center;">
            <div style="font-size:2rem; font-weight:700; color:#ff4b4b;
                        line-height:1; margin-bottom:0.4rem;">{attack_flows:,}</div>
            <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase;
                        letter-spacing:0.8px;">Attack Flows Detected</div>
          </div>
          <div style="flex:1; background:#161b22; border:1px solid #30363d;
                      border-radius:10px; padding:1.2rem 1.5rem; text-align:center;">
            <div style="font-size:2rem; font-weight:700; color:#3fb950;
                        line-height:1; margin-bottom:0.4rem;">{benign_flows:,}</div>
            <div style="font-size:0.8rem; color:#8b949e; text-transform:uppercase;
                        letter-spacing:0.8px;">Benign Flows</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Results table
        results_df = pd.DataFrame({
            "Row":             range(1, total_flows + 1),
            "Predicted Class": pred_labels,
            "Confidence":      [f"{c:.2%}" for c in confidences],
        })

        MAX_STYLED_ROWS = 50000
        if len(results_df) <= MAX_STYLED_ROWS:
            pd.set_option(
                "styler.render.max_elements",
                len(results_df) * len(results_df.columns)
            )
            styled = results_df.style.apply(highlight_attacks, axis=1)
            st.dataframe(styled, use_container_width=True,
                         height=380, hide_index=True)
        else:
            st.info(
                f"File has {total_flows:,} rows. "
                "Row highlighting is disabled for large files to maintain performance."
            )
            st.dataframe(results_df, use_container_width=True,
                         height=380, hide_index=True)

        # Bar chart
        class_counts = (pd.Series(pred_labels)
                        .value_counts()
                        .reset_index())
        class_counts.columns = ["Class", "Count"]
        bar_colors = [
            "#00d4ff" if c == "BENIGN" else "#ff4b4b"
            for c in class_counts["Class"]
        ]
        fig = go.Figure(go.Bar(
            x=class_counts["Count"],
            y=class_counts["Class"],
            orientation="h",
            marker_color=bar_colors,
            text=class_counts["Count"],
            textposition="outside",
            textfont=dict(color="#ffffff"),
            hovertemplate="%{y}: %{x} flows<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font_color="#ffffff",
            title=dict(
                text="Predicted Traffic Class Distribution",
                font=dict(size=15, color="#ffffff"),
            ),
            xaxis=dict(
                title="Flow Count",
                gridcolor="#21262d",
                tickfont=dict(color="#8b949e"),
            ),
            yaxis=dict(
                tickfont=dict(color="#ffffff"),
                categoryorder="total ascending",
            ),
            margin=dict(l=20, r=60, t=50, b=20),
            height=max(300, len(class_counts) * 55 + 100),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Download button right-aligned
        csv_out = results_df.to_csv(index=False).encode("utf-8")
        _, col_dl = st.columns([3, 1])
        with col_dl:
            st.download_button(
                label="⬇ Download Results as CSV",
                data=csv_out,
                file_name="nids_results.csv",
                mime="text/csv",
            )

    render_footer()


def render_footer():
    st.markdown(
        "<div style='text-align:center; font-size:0.78rem; color:#8b949e; "
        "margin-top:3rem; padding-top:1rem; "
        "border-top:1px solid #21262d;'>"
        "Jesudun Temiloluwa Esther &nbsp;|&nbsp; 22/10331 &nbsp;|&nbsp; "
        "Caleb University &nbsp;|&nbsp; Final Year Project 2026"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()