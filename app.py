import os
import sys
import uuid
import hashlib
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import seaborn as sns
except Exception:  # pragma: no cover - app still works without seaborn
    sns = None


REQUIRED_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

NUMERIC_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]

MODEL_CANDIDATES = [
    Path("saved_models/final_churn_model.pkl"),
    Path("saved_models/best_churn_model.pkl"),
]

DATA_CANDIDATES = [
    Path("data/Churn_Modelling.csv"),
    Path("Churn_Modelling.csv"),
]

RESULT_FILES = {
    "Tuned Results": Path("saved_models/tuned_results.csv"),
    "Balanced Results": Path("saved_models_balanced/balanced_results.csv"),
    "Undersampled Results": Path("saved_models_undersampled/undersampled_results.csv"),
}

HISTORY_PATH = Path("data/prediction_history.csv")
HISTORY_COLUMNS = [
    "Prediction_ID",
    "Timestamp",
    "Prediction_Source",
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Churn_Prediction",
    "Churn_Label",
    "Churn_Probability",
    "Risk_Level",
]


def add_churn_features(X):
    X = X.copy()

    # Product-based features
    X["OneProduct"] = (X["NumOfProducts"] == 1).astype(int)
    X["TwoProducts"] = (X["NumOfProducts"] == 2).astype(int)
    X["ThreePlusProducts"] = (X["NumOfProducts"] >= 3).astype(int)

    # Balance-based features
    X["BalanceZero"] = (X["Balance"] == 0).astype(int)
    X["HasBalance"] = (X["Balance"] > 0).astype(int)
    X["BalanceSalaryRatio"] = X["Balance"] / (X["EstimatedSalary"] + 1)

    # Interaction features
    X["Age_IsActive"] = X["Age"] * X["IsActiveMember"]
    X["Age_NumProducts"] = X["Age"] * X["NumOfProducts"]
    X["Germany_Age"] = ((X["Geography"] == "Germany").astype(int)) * X["Age"]

    # Age group feature
    X["AgeGroup"] = pd.cut(
        X["Age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["Young", "Adult", "MiddleAge", "Senior", "Old"],
        include_lowest=True
    ).astype(str)

    return X


# Notebook-saved FunctionTransformer objects often reference __main__.
# Register the function there before joblib loads the pipeline.
setattr(sys.modules["__main__"], "add_churn_features", add_churn_features)


st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
        """
        <style>
            :root {
                --bg: #0f1117;
                --bg2: #151924;
                --card: #1b1f2a;
                --border: rgba(220, 189, 146, 0.25);
                --text: #eae4dc;
                --muted: #b8b0a4;
                --accent: #dcbd92;
                --highlight: #f8bb38;
                --danger: #ff5c5c;
                --success: #3ddc97;
            }

            .stApp {
                background: radial-gradient(circle at top left, rgba(220,189,146,.08), transparent 32%),
                            var(--bg);
                color: var(--text);
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #151924 0%, #10131b 100%);
                border-right: 1px solid var(--border);
            }

            [data-testid="stSidebar"] * { color: var(--text); }
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
                color: var(--accent);
                text-transform: uppercase;
                letter-spacing: .08em;
            }

            h1, h2, h3, h4, h5, h6 { color: var(--text); letter-spacing: 0; }
            p, li, label, span, div { color: var(--text); }

            .main-title {
                font-size: 48px;
                font-weight: 850;
                text-align: center;
                text-transform: uppercase;
                letter-spacing: 2px;
                color: var(--text);
                margin: 14px 0 6px;
            }

            .subtitle {
                max-width: 920px;
                margin: 0 auto 28px;
                text-align: center;
                color: var(--muted);
                font-size: 17px;
                line-height: 1.7;
            }

            .section-header {
                text-align: center;
                margin: 28px 0 22px;
            }

            .section-header h2 {
                color: var(--text);
                font-size: 36px;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 4px;
                margin-bottom: 8px;
            }

            .section-header .accent-line {
                height: 1px;
                width: min(520px, 70%);
                margin: 10px auto;
                background: linear-gradient(90deg, transparent, var(--accent), transparent);
            }

            .section-header p {
                color: var(--muted);
                font-size: 14px;
                letter-spacing: .03em;
                margin-top: 8px;
            }

            .eda-header {
                text-align: center;
                margin: 42px 0 28px 0;
            }

            .eda-header h2 {
                color: var(--text);
                font-size: 36px;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 4px;
                margin-bottom: 10px;
            }

            .eda-header .accent-line {
                width: 100px;
                height: 4px;
                background: linear-gradient(90deg, transparent, var(--accent), transparent);
                margin: 0 auto;
                border-radius: 2px;
            }

            .eda-subtitle {
                color: var(--highlight);
                font-size: 14px;
                font-weight: 500;
                letter-spacing: 2px;
                text-transform: uppercase;
                opacity: .8;
                margin-top: 12px;
            }

            .premium-card, .insight-card, .metric-card {
                background: linear-gradient(180deg, rgba(255,255,255,.025), rgba(255,255,255,.005)), var(--card);
                border: 1px solid var(--border);
                border-radius: 8px;
                box-shadow: 0 16px 40px rgba(0,0,0,.22);
            }

            .premium-card {
                padding: 22px;
                min-height: 145px;
                margin-bottom: 16px;
            }

            .premium-card h3 {
                color: var(--accent);
                font-size: 16px;
                text-transform: uppercase;
                letter-spacing: .08em;
                margin: 0 0 10px;
            }

            .premium-card p, .insight-card p {
                color: var(--muted);
                line-height: 1.65;
                margin: 0;
            }

            .eda-card {
                background: linear-gradient(135deg, rgba(27,31,42,.96), rgba(15,17,23,.96));
                border: 1px solid rgba(220,189,146,.28);
                border-radius: 8px;
                padding: 22px 26px;
                margin: 18px 0 26px 0;
                color: var(--text);
                box-shadow: 0 16px 40px rgba(0,0,0,.22);
            }

            .eda-card h3 {
                color: var(--accent);
                margin-top: 0;
                font-size: 19px;
            }

            .eda-card p, .eda-card li {
                color: var(--muted);
                line-height: 1.7;
            }

            .metric-card {
                padding: 18px;
                min-height: 112px;
                margin-bottom: 14px;
            }

            .metric-label {
                color: var(--muted);
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: .11em;
                margin-bottom: 8px;
            }

            .metric-value {
                color: var(--text);
                font-size: 30px;
                font-weight: 820;
                line-height: 1.1;
                overflow-wrap: anywhere;
            }

            .insight-card {
                padding: 16px 18px;
                border-left: 3px solid var(--accent);
                margin: 12px 0 18px;
            }

            .risk-low, .risk-medium, .risk-high, .risk-neutral {
                border-radius: 8px;
                padding: 16px 18px;
                border: 1px solid var(--border);
                font-weight: 750;
                margin: 14px 0;
            }

            .risk-low { background: rgba(61, 220, 151, .12); color: var(--success); }
            .risk-medium { background: rgba(248, 187, 56, .13); color: var(--highlight); }
            .risk-high { background: rgba(255, 92, 92, .12); color: var(--danger); }
            .risk-neutral { background: rgba(220, 189, 146, .10); color: var(--accent); }

            div.stButton > button, div.stDownloadButton > button {
                background: linear-gradient(135deg, var(--accent), var(--highlight));
                color: #111318;
                border: 0;
                border-radius: 8px;
                font-weight: 800;
                padding: .65rem 1.2rem;
            }

            div.stButton > button:hover, div.stDownloadButton > button:hover {
                border: 0;
                color: #111318;
                filter: brightness(1.04);
            }

            [data-testid="stMetric"] {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 8px;
                padding: 16px;
            }

            [data-testid="stMetricLabel"] p { color: var(--muted); }
            [data-testid="stMetricValue"] { color: var(--text); }

            .stDataFrame {
                border: 1px solid var(--border);
                border-radius: 8px;
                overflow: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=None):
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-header">
            <h2>{title}</h2>
            <div class="accent-line"></div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(title, body):
    st.markdown(
        f"""
        <div class="premium-card">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight(text):
    st.markdown(f'<div class="insight-card"><p>{text}</p></div>', unsafe_allow_html=True)


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def find_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_resource(show_spinner=False)
def load_model():
    model_path = find_existing_path(MODEL_CANDIDATES)
    if model_path is None:
        return None, None, "No model file found in saved_models."
    try:
        return joblib.load(model_path), model_path, None
    except Exception as exc:
        return None, model_path, str(exc)


@st.cache_data(show_spinner=False)
def load_dataset():
    data_path = find_existing_path(DATA_CANDIDATES)
    if data_path is None:
        return None, None, "Dataset not found. Expected data/Churn_Modelling.csv."
    try:
        return pd.read_csv(data_path), data_path, None
    except Exception as exc:
        return None, data_path, str(exc)


def yes_no_to_int(value):
    return 1 if value == "Yes" else 0


def normalize_binary_columns(df):
    df = df.copy()
    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "t": 1,
        "1": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "f": 0,
        "0": 0,
    }
    for column in ["HasCrCard", "IsActiveMember"]:
        if column in df.columns:
            normalized = df[column].astype(str).str.strip().str.lower().map(mapping)
            df[column] = normalized.where(normalized.notna(), df[column])
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def prepare_prediction_frame(df):
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        return None, missing, []

    extra = [column for column in df.columns if column not in REQUIRED_COLUMNS]
    prepared = normalize_binary_columns(df[REQUIRED_COLUMNS])
    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared, [], extra


def empty_prediction_history():
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def load_prediction_history():
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_PATH.exists():
        history = empty_prediction_history()
        history.to_csv(HISTORY_PATH, index=False)
        return history

    try:
        history = pd.read_csv(HISTORY_PATH)
    except pd.errors.EmptyDataError:
        history = empty_prediction_history()
    except Exception:
        return empty_prediction_history()

    for column in HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = np.nan
    return history[HISTORY_COLUMNS]


def save_prediction_to_history(prediction_df):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_rows = prediction_df.copy()
    for column in HISTORY_COLUMNS:
        if column not in new_rows.columns:
            new_rows[column] = np.nan
    new_rows = new_rows[HISTORY_COLUMNS]

    history = load_prediction_history()
    updated = pd.concat([history, new_rows], ignore_index=True)
    updated.to_csv(HISTORY_PATH, index=False)


def clear_prediction_history():
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    empty_prediction_history().to_csv(HISTORY_PATH, index=False)


def make_prediction_ids(count):
    return [uuid.uuid4().hex for _ in range(count)]


def current_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def uploaded_file_key(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    digest = hashlib.sha256(file_bytes).hexdigest()
    return f"{uploaded_file.name}:{len(file_bytes)}:{digest}"


def get_churn_probability(model, data):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()
    return None


def get_decision_score(model, data):
    if hasattr(model, "decision_function"):
        scores = model.decision_function(data)
        return np.asarray(scores).ravel()
    return None


def risk_level(probability):
    if probability is None or pd.isna(probability):
        return "Unavailable"
    if probability < 0.35:
        return "Low Risk"
    if probability < 0.65:
        return "Medium Risk"
    return "High Risk"


def risk_class(level):
    return {
        "Low Risk": "risk-low",
        "Medium Risk": "risk-medium",
        "High Risk": "risk-high",
    }.get(level, "risk-neutral")


def add_eda_features(df):
    df = df.copy()
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 30, 40, 50, 60, 100],
        labels=["Young", "Adult", "MiddleAge", "Senior", "Old"],
        include_lowest=True,
    ).astype(str)
    df["HasBalance"] = (df["Balance"] > 0).astype(int)
    df["BalanceZero"] = (df["Balance"] == 0).astype(int)
    return df


def themed_fig(width=8, height=4.6):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#151924")
    ax.tick_params(colors="#b8b0a4")
    ax.xaxis.label.set_color("#eae4dc")
    ax.yaxis.label.set_color("#eae4dc")
    for spine in ax.spines.values():
        spine.set_color("#413a35")
    return fig, ax


def bar_chart(series, title, xlabel="", ylabel="", color="#dcbd92"):
    fig, ax = themed_fig()
    series.plot(kind="bar", ax=ax, color=color, edgecolor="#f8bb38", linewidth=.7)
    ax.set_title(title, color="#dcbd92", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.18)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def grouped_churn_chart(df, column, title):
    grouped = pd.crosstab(df[column], df["Exited"], normalize="index") * 100
    grouped = grouped.rename(columns={0: "Not Exited", 1: "Exited"})
    fig, ax = themed_fig()
    grouped.plot(kind="bar", ax=ax, color=["#3ddc97", "#dcbd92"], edgecolor="#151924")
    ax.set_title(title, color="#dcbd92", fontsize=15, fontweight="bold", pad=14)
    ax.set_ylabel("Customer share (%)")
    ax.set_xlabel(column)
    ax.legend(facecolor="#1b1f2a", edgecolor="#413a35", labelcolor="#eae4dc")
    ax.grid(axis="y", alpha=.18)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def notebook_churn_rate_table(data, column):
    table = (
        data.groupby(column)["Exited"]
        .agg(Count="count", Churn_Rate="mean")
        .reset_index()
        .sort_values("Churn_Rate", ascending=False)
    )
    table["Churn_Rate_%"] = (table["Churn_Rate"] * 100).round(2)
    return table


def notebook_plot_churn_rate(data, column, title=None, rotate=0):
    table = notebook_churn_rate_table(data, column)
    fig, ax = themed_fig(width=9, height=5)
    ax.bar(table[column].astype(str), table["Churn_Rate_%"], color="#dcbd92", edgecolor="#f8bb38", linewidth=.7)
    ax.set_title(title or f"Churn Rate by {column}", color="#dcbd92", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel(column)
    ax.set_ylabel("Churn Rate (%)")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=rotate)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.dataframe(table, use_container_width=True, hide_index=True)


def home_page(model_path, data, data_path):
    st.markdown('<div class="main-title">Bank Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">A premium Streamlit interface for predicting customer churn with the saved scikit-learn pipeline from the notebook. The app preserves the notebook feature engineering contract and never retrains the model.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        card("Binary Classification", "Predict whether a customer belongs to the not exited or exited class.")
    with col2:
        card("Target: Exited", "The model estimates the likelihood that a customer leaves the bank.")
    with col3:
        card("Algorithms: 5 Models", "Logistic Regression, KNN, Decision Tree, Random Forest, and SVC were tested.")
    with col4:
        final_model = model_path.as_posix() if model_path else "Not loaded"
        card("Final Model", f"Loaded production pipeline: {final_model}")

    section_header("Dataset Snapshot", "Available training data metrics")
    if data is None:
        st.warning("Dataset not found. Add data/Churn_Modelling.csv to show dataset metrics.")
        return

    features = [column for column in data.columns if column != "Exited"]
    churn_rate = data["Exited"].mean() if "Exited" in data.columns else np.nan
    dist = data["Exited"].value_counts().sort_index() if "Exited" in data.columns else pd.Series(dtype=int)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Customers", f"{len(data):,}")
    with col2:
        metric_card("Churn Rate", "N/A" if pd.isna(churn_rate) else f"{churn_rate:.1%}")
    with col3:
        metric_card("Features", f"{len(features):,}")
    with col4:
        metric_card("Class Distribution", " | ".join(f"{idx}: {val:,}" for idx, val in dist.items()))

    st.caption(f"Dataset source: {data_path}")


def single_customer_prediction_page(model):
    section_header("Single Customer Prediction", "Score an individual customer with the saved pipeline")

    with st.form("single_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.number_input("CreditScore", min_value=300, max_value=900, value=650)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
        with col2:
            balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=75000.0, step=1000.0)
            num_products = st.selectbox("NumOfProducts", [1, 2, 3, 4])
            has_cr_card = st.selectbox("HasCrCard", ["Yes", "No"])
            is_active_member = st.selectbox("IsActiveMember", ["Yes", "No"])
            estimated_salary = st.number_input("EstimatedSalary", min_value=0.0, max_value=250000.0, value=100000.0, step=1000.0)

        submitted = st.form_submit_button("Predict")

    input_df = pd.DataFrame(
        [[
            credit_score,
            geography,
            gender,
            age,
            tenure,
            balance,
            num_products,
            yes_no_to_int(has_cr_card),
            yes_no_to_int(is_active_member),
            estimated_salary,
        ]],
        columns=REQUIRED_COLUMNS,
    )

    st.markdown("#### Customer Input Preview")
    st.dataframe(input_df, use_container_width=True, hide_index=True)

    if not submitted:
        return

    try:
        prediction = int(model.predict(input_df)[0])
        probabilities = get_churn_probability(model, input_df)
        score = get_decision_score(model, input_df) if probabilities is None else None
        probability = None if probabilities is None else float(probabilities[0])
        level = risk_level(probability)
        label = "Exited" if prediction == 1 else "Not Exited"

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Prediction", label)
        with col2:
            metric_card("Churn Probability", "Unavailable" if probability is None else f"{probability:.1%}")
        with col3:
            metric_card("Risk Level", level)

        if probability is not None:
            message = f"Prediction: {label}. Estimated churn probability is {probability:.1%}, classified as {level}."
        elif score is not None:
            message = f"Prediction: {label}. Probability is unavailable; decision score is {float(score[0]):.4f}."
        else:
            message = f"Prediction: {label}. Probability and decision score are unavailable for this model."

        st.markdown(f'<div class="{risk_class(level)}">{message}</div>', unsafe_allow_html=True)
        if level == "Low Risk":
            st.success("Low Risk: this customer profile is comparatively stable.")
        elif level == "Medium Risk":
            st.warning("Medium Risk: monitor this customer and consider retention actions.")
        elif level == "High Risk":
            st.error("High Risk: this customer is likely to churn and should be prioritized.")

        history_row = input_df.copy()
        history_row.insert(0, "Prediction_ID", make_prediction_ids(1))
        history_row.insert(1, "Timestamp", current_timestamp())
        history_row.insert(2, "Prediction_Source", "Single")
        history_row["Churn_Prediction"] = prediction
        history_row["Churn_Label"] = label
        history_row["Churn_Probability"] = np.nan if probability is None else probability
        history_row["Risk_Level"] = level
        save_prediction_to_history(history_row)
        st.success("Prediction saved to history.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


def batch_csv_prediction_page(model):
    section_header("Batch CSV Prediction", "Upload a customer CSV and export scored predictions")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        insight("Required columns: " + ", ".join(REQUIRED_COLUMNS))
        return

    batch_key = uploaded_file_key(uploaded_file)

    try:
        data = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV file: {exc}")
        return

    prediction_data, missing, extra = prepare_prediction_frame(data)
    if missing:
        st.error(f"Missing required columns: {missing}")
        return
    if extra:
        st.info(f"Ignoring extra columns for prediction: {extra}")

    if prediction_data[NUMERIC_COLUMNS].isna().any().any():
        bad_columns = prediction_data[NUMERIC_COLUMNS].columns[prediction_data[NUMERIC_COLUMNS].isna().any()].tolist()
        st.error(f"These numeric/binary columns contain invalid values after conversion: {bad_columns}")
        return

    try:
        predictions = model.predict(prediction_data).astype(int)
        probabilities = get_churn_probability(model, prediction_data)
        output = data.copy()
        output["Churn_Prediction"] = predictions
        output["Churn_Label"] = np.where(predictions == 1, "Exited", "Not Exited")
        output["Churn_Probability"] = np.nan if probabilities is None else probabilities
        output["Risk_Level"] = output["Churn_Probability"].apply(risk_level)

        history_rows = prediction_data.copy()
        history_rows.insert(0, "Prediction_ID", make_prediction_ids(len(history_rows)))
        history_rows.insert(1, "Timestamp", current_timestamp())
        history_rows.insert(2, "Prediction_Source", "Batch")
        history_rows["Churn_Prediction"] = predictions
        history_rows["Churn_Label"] = np.where(predictions == 1, "Exited", "Not Exited")
        history_rows["Churn_Probability"] = np.nan if probabilities is None else probabilities
        history_rows["Risk_Level"] = history_rows["Churn_Probability"].apply(risk_level)

        saved_batch_keys = st.session_state.setdefault("saved_batch_history_keys", set())
        batch_history_key = f"{batch_key}:{len(history_rows)}"
        batch_saved_now = False
        if batch_history_key not in saved_batch_keys:
            save_prediction_to_history(history_rows)
            saved_batch_keys.add(batch_history_key)
            batch_saved_now = True

        total = len(output)
        churned = int((predictions == 1).sum())
        churn_rate = churned / total if total else 0
        avg_prob = output["Churn_Probability"].mean()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Total Customers", f"{total:,}")
        with col2:
            metric_card("Predicted Churned", f"{churned:,}")
        with col3:
            metric_card("Predicted Churn Rate", f"{churn_rate:.1%}")
        with col4:
            metric_card("Avg Churn Probability", "N/A" if pd.isna(avg_prob) else f"{avg_prob:.1%}")

        st.dataframe(output.head(100), use_container_width=True, hide_index=True)
        csv = output.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", csv, "churn_predictions.csv", "text/csv")
        if batch_saved_now:
            st.success("Batch predictions saved to history.")
        else:
            st.info("This uploaded batch is already saved to history.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


def eda_page(data):
    st.markdown(
        """
        <div class="eda-header">
            <h2>4. Analyse Exploratoire des Données</h2>
            <div class="accent-line"></div>
            <div class="eda-subtitle">EDA aligned with the future Feature Engineering</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if data is None:
        st.warning("Dataset not found. Add data/Churn_Modelling.csv to enable the EDA dashboard.")
        return
    if "Exited" not in data.columns:
        st.error("The EDA page requires an Exited target column.")
        return

    df = data.copy()

    st.markdown(
        """
        <div class="eda-card">
        <h3>Logique de cette partie</h3>

        <p>Cette partie EDA n'est pas seulement descriptive. Elle sert à comprendre les variables qui influencent le churn et à préparer le <strong>Feature Engineering</strong>.</p>
        <p>Chaque observation importante sera ensuite transformée en une variable explicative utile pour les modèles.</p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    target_counts = df["Exited"].value_counts().sort_index()
    target_percent = (df["Exited"].value_counts(normalize=True).sort_index() * 100).round(2)
    target_summary = pd.DataFrame({
        "Class": ["Not Exited", "Exited"],
        "Count": target_counts.values,
        "Percentage": target_percent.values,
    })
    st.dataframe(target_summary, use_container_width=True, hide_index=True)

    fig, ax = themed_fig(width=7, height=5)
    ax.bar(["Not Exited", "Exited"], target_counts.values, color="#dcbd92", edgecolor="#f8bb38", linewidth=.7)
    ax.set_title("Target Distribution - Exited", color="#dcbd92", fontsize=15, fontweight="bold", pad=14)
    ax.set_ylabel("Number of customers")
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        """
        <div class="eda-card">
        <h3>Observation EDA — Target imbalance</h3>

        <p>La variable cible <strong>Exited</strong> est déséquilibrée : la majorité des clients ne quittent pas la banque.</p>
        <p>Conséquence directe : l'<strong>accuracy</strong> seule ne suffit pas. Dans l'évaluation finale, on utilisera aussi <strong>Recall, F1-score, Balanced Accuracy et ROC-AUC</strong>.</p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="eda-header">
            <h2>4.1 Categorical Analysis</h2>
            <div class="accent-line"></div>
            <div class="eda-subtitle">Geography, Gender, Products, Credit Card and Activity</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    notebook_plot_churn_rate(df, "Geography", "Churn Rate by Geography")
    st.markdown(
        """
        <div class="eda-card">
        <h3>Observation — Geography</h3>

        <p>Les clients en <strong>Germany</strong> présentent généralement un taux de churn plus élevé que les autres pays.</p>
        <p>Cette observation justifie la création d'une variable d'interaction <strong>Germany_Age</strong>, car l'effet du pays peut être renforcé par l'âge.</p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    notebook_plot_churn_rate(df, "Gender", "Churn Rate by Gender")
    notebook_plot_churn_rate(df, "NumOfProducts", "Churn Rate by Number of Products")
    st.markdown(
        """
        <div class="eda-card">
        <h3>Observation — NumOfProducts</h3>

        <p>Les clients ayant <strong>3 ou 4 produits</strong> ont un comportement très différent et un taux de churn très élevé.</p>
        <p>Cette observation justifie la création de la variable <strong>ThreePlusProducts</strong>.</p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    notebook_plot_churn_rate(df, "IsActiveMember", "Churn Rate by Active Membership")
    st.markdown(
        """
        <div class="eda-card">
        <h3>Observation — Activity</h3>

        <p>Les clients inactifs quittent plus souvent la banque que les clients actifs.</p>
        <p>L'activité du client devient encore plus intéressante lorsqu'elle est combinée avec l'âge. Cela justifie la variable <strong>Age_IsActive</strong>.</p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    notebook_plot_churn_rate(df, "HasCrCard", "Churn Rate by Credit Card Ownership")

    st.markdown(
        """
        <div class="eda-header">
            <h2>4.2 Numerical Analysis</h2>
            <div class="accent-line"></div>
            <div class="eda-subtitle">Age, Balance, Salary and Credit Score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if sns is None:
        st.error("Seaborn is required to reproduce the notebook KDE and heatmap charts exactly.")
        return

    numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
    for col in numeric_cols:
        fig, ax = themed_fig(width=9, height=5)
        sns.kdeplot(data=df, x=col, hue="Exited", fill=True, common_norm=False, alpha=0.35, ax=ax, palette=["#3ddc97", "#dcbd92"])
        ax.set_title(f"Distribution of {col} by Exited", color="#dcbd92", fontsize=15, fontweight="bold", pad=14)
        ax.grid(alpha=0.25)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    age_bins = [17, 30, 40, 50, 60, 100]
    age_labels = ["18-30", "31-40", "41-50", "51-60", "60+"]

    df_eda = df.copy()
    df_eda["AgeGroup"] = pd.cut(
        df_eda["Age"],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True,
    )

    notebook_plot_churn_rate(df_eda, "AgeGroup", "Churn Rate by Age Group")
    st.markdown(
        """
        <div class="eda-card">
        <h3>Observation — Age</h3>

        <p>L'âge est l'une des variables les plus importantes. Les clients plus âgés, surtout certaines tranches, présentent un risque de churn plus élevé.</p>
        <p>Cette observation justifie deux variables :</p>
        <ul>
        <li><strong>AgeGroup</strong> : transformation de l'âge en catégories interprétables.</li>
        <li><strong>Age_NumProducts</strong> : interaction entre l'âge et le nombre de produits.</li>
        </ul>

        </div>
        """,
        unsafe_allow_html=True,
    )

    df_eda["BalanceZero"] = (df_eda["Balance"] == 0).astype(int)
    df_eda["HasBalance"] = (df_eda["Balance"] > 0).astype(int)

    notebook_plot_churn_rate(df_eda, "BalanceZero", "Churn Rate by Zero Balance")
    notebook_plot_churn_rate(df_eda, "HasBalance", "Churn Rate by Has Balance")
    st.markdown(
        """
        <div class="eda-card">
        <h3>Observation — Balance</h3>

        <p>Le comportement des clients avec un solde nul est différent de celui des clients ayant un solde positif.</p>
        <p>Cette observation justifie la création de <strong>BalanceZero</strong> et <strong>HasBalance</strong>.</p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    df_eda["BalanceSalaryRatio"] = df_eda["Balance"] / (df_eda["EstimatedSalary"] + 1)
    df_eda["BalanceSalaryRatioGroup"] = pd.qcut(
        df_eda["BalanceSalaryRatio"],
        q=5,
        duplicates="drop",
    )

    ratio_table = notebook_churn_rate_table(df_eda, "BalanceSalaryRatioGroup")
    st.dataframe(ratio_table, use_container_width=True, hide_index=True)

    fig, ax = themed_fig(width=10, height=5)
    ax.bar(ratio_table["BalanceSalaryRatioGroup"].astype(str), ratio_table["Churn_Rate_%"], color="#dcbd92", edgecolor="#f8bb38", linewidth=.7)
    ax.set_title("Churn Rate by Balance / Salary Ratio Quantiles", color="#dcbd92", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Balance / Salary Ratio group")
    ax.set_ylabel("Churn Rate (%)")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=35, ha="right")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        """
        <div class="eda-card">
        <h3>Observation — BalanceSalaryRatio</h3>

        <p>Le solde seul ne donne pas toute l'information. Le rapport entre le solde et le salaire estimé permet de mieux représenter le poids financier du solde pour chaque client.</p>
        <p>Cette observation justifie la création de <strong>BalanceSalaryRatio</strong>.</p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    corr_cols = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"
    ]

    corr = df[corr_cols].corr()

    fig, ax = themed_fig(width=10, height=7)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="copper", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap", color="#dcbd92", fontsize=15, fontweight="bold", pad=14)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        """
        <div class="eda-header">
            <h2>4.3 EDA to Feature Engineering Mapping</h2>
            <div class="accent-line"></div>
            <div class="eda-subtitle">Every engineered feature is justified by the EDA</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eda_to_fe = pd.DataFrame({
        "EDA observation": [
            "Customers with zero balance behave differently from customers with positive balance.",
            "Balance alone is less informative than balance compared to estimated salary.",
            "Age has a strong relationship with churn.",
            "Customer activity impacts churn and interacts with age.",
            "Number of products has a non-linear impact; 3+ products are high-risk.",
            "Germany has a higher churn rate than other countries.",
            "Age groups show different churn patterns."
        ],
        "Feature created": [
            "BalanceZero, HasBalance",
            "BalanceSalaryRatio",
            "AgeGroup",
            "Age_IsActive",
            "ThreePlusProducts, Age_NumProducts",
            "Germany_Age",
            "AgeGroup"
        ],
        "Expected benefit": [
            "Capture the special behavior of customers with no balance.",
            "Normalize balance according to customer's income level.",
            "Capture non-linear age effects.",
            "Represent combined effect of age and activity.",
            "Capture non-linear product risk.",
            "Capture interaction between geography and age.",
            "Make age more interpretable for models."
        ]
    })

    st.dataframe(eda_to_fe, use_container_width=True, hide_index=True)


def load_results(path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def normalize_metric_name(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def results_summary(label, df):
    st.markdown(f"#### {label}")
    st.dataframe(df, use_container_width=True, hide_index=True)

    model_col = normalize_metric_name(df, ["Model", "model"])
    accuracy_col = normalize_metric_name(df, ["Accuracy", "accuracy", "Best_CV_Accuracy"])
    auc_col = normalize_metric_name(df, ["AUC", "auc", "ROC_AUC"])
    f1_col = normalize_metric_name(df, ["F1_churn", "F1", "F1_macro"])
    recall_col = normalize_metric_name(df, ["Recall_churn", "Recall", "Recall_macro"])

    if model_col and accuracy_col:
        best_idx = df[accuracy_col].astype(float).idxmax()
        best = df.loc[best_idx]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Best Model", best[model_col])
        with col2:
            metric_card("Accuracy", f"{float(best[accuracy_col]):.3f}")
        with col3:
            metric_card("AUC", "N/A" if auc_col is None else f"{float(best[auc_col]):.3f}")
        with col4:
            value = recall_col if recall_col else f1_col
            metric_card("Recall/F1 Churn", "N/A" if value is None else f"{float(best[value]):.3f}")

    for metric in [accuracy_col, auc_col, recall_col]:
        if model_col and metric:
            fig, ax = themed_fig()
            chart_df = df[[model_col, metric]].copy()
            chart_df[metric] = pd.to_numeric(chart_df[metric], errors="coerce")
            chart_df.dropna().plot(kind="bar", x=model_col, y=metric, ax=ax, color="#dcbd92", legend=False)
            ax.set_title(f"{metric} by Model", color="#dcbd92", fontsize=15, fontweight="bold", pad=14)
            ax.set_xlabel("Model")
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=.18)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def model_performance_page():
    section_header("Model Performance", "Compare saved notebook result files when available")
    any_results = False
    for label, path in RESULT_FILES.items():
        df = load_results(path)
        if df is None:
            st.info(f"{label} not found at {path}.")
            continue
        any_results = True
        results_summary(label, df)

    if not any_results:
        st.warning("No saved result CSV files were found.")

    insight("RandomOverSampler can improve churn recall, but it often decreases overall accuracy. For this project, selecting the final model without oversampling is reasonable when the main objective is accuracy.")
    insight("An accuracy around 0.86 is realistic for this dataset. Reaching 0.90 cleanly is difficult without richer behavioral, transaction, or customer-interaction variables.")


def prediction_history_page():
    section_header("Prediction History", "Local audit trail for single and batch churn predictions")

    history = load_prediction_history()
    if history.empty:
        st.info("No predictions have been saved yet. Make a single or batch prediction to populate this page.")
    else:
        history["Churn_Probability"] = pd.to_numeric(history["Churn_Probability"], errors="coerce")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        source_filter = st.selectbox("Prediction Source", ["All", "Single", "Batch"])
    with filter_col2:
        risk_filter = st.selectbox("Risk Level", ["All", "Low Risk", "Medium Risk", "High Risk"])
    with filter_col3:
        label_filter = st.selectbox("Churn Label", ["All", "Not Exited", "Exited"])

    filtered = history.copy()
    if source_filter != "All":
        filtered = filtered[filtered["Prediction_Source"] == source_filter]
    if risk_filter != "All":
        filtered = filtered[filtered["Risk_Level"] == risk_filter]
    if label_filter != "All":
        filtered = filtered[filtered["Churn_Label"] == label_filter]

    total_predictions = len(filtered)
    single_predictions = int((filtered["Prediction_Source"] == "Single").sum()) if not filtered.empty else 0
    batch_predictions = int((filtered["Prediction_Source"] == "Batch").sum()) if not filtered.empty else 0
    churned_customers = int((filtered["Churn_Label"] == "Exited").sum()) if not filtered.empty else 0
    avg_probability = filtered["Churn_Probability"].mean() if not filtered.empty else np.nan

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Total Predictions", f"{total_predictions:,}")
    with col2:
        metric_card("Single Predictions", f"{single_predictions:,}")
    with col3:
        metric_card("Batch Predictions", f"{batch_predictions:,}")
    with col4:
        metric_card("Predicted Churned Customers", f"{churned_customers:,}")
    with col5:
        metric_card("Average Churn Probability", "N/A" if pd.isna(avg_probability) else f"{avg_probability:.1%}")

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Filtered History CSV",
        csv,
        "filtered_prediction_history.csv",
        "text/csv",
        disabled=filtered.empty,
    )

    st.markdown("---")
    st.markdown("#### Clear Prediction History")
    confirm = st.checkbox("I confirm that I want to delete the prediction history")
    if st.button("Clear Prediction History"):
        if confirm:
            clear_prediction_history()
            st.success("Prediction history cleared.")
            st.rerun()
        else:
            st.warning("Please confirm before clearing the prediction history.")


def model_information_page(model_path):
    section_header("Model Information", "What the app loads, predicts, and fixes")
    col1, col2 = st.columns(2)
    with col1:
        card("Problem", "Bank Customer Churn Prediction")
        card("Type", "Binary Classification")
        card("Target", "Exited")
        card("Final Model Path", model_path.as_posix() if model_path else "Not loaded")
    with col2:
        card("Algorithms Tested", "Logistic Regression, KNN, Decision Tree, Random Forest, and SVC.")
        card("Pipeline Design", "Preprocessing and feature engineering are inside the saved sklearn pipeline.")
        card("Class Imbalance", "The notebook analyzed imbalance because churn customers are the minority class.")
        card("No Retraining", "The Streamlit app only loads the saved model and predicts.")

    section_header("Input Features")
    st.markdown(", ".join(f"`{column}`" for column in REQUIRED_COLUMNS))

    section_header("Engineered Features")
    engineered = [
        "OneProduct",
        "TwoProducts",
        "ThreePlusProducts",
        "BalanceZero",
        "HasBalance",
        "BalanceSalaryRatio",
        "Age_IsActive",
        "Age_NumProducts",
        "Germany_Age",
        "AgeGroup",
    ]
    st.markdown(", ".join(f"`{feature}`" for feature in engineered))

    section_header("Why The Previous Error Happened")
    insight("The previous app failed because the saved pipeline expected engineered columns OneProduct and TwoProducts, but the Streamlit app's add_churn_features function did not create them. The new app fixes this by defining the exact same feature engineering function before loading the saved model.")


def sidebar_menu(model_path):
    with st.sidebar:
        st.markdown("## Churn Studio")
        st.caption("Premium ML dashboard")
        st.markdown("---")
        page = st.radio(
            "Navigation",
            [
                "Home",
                "Single Customer Prediction",
                "Batch CSV Prediction",
                "Prediction History",
                "Exploratory Data Analysis",
                "Model Performance",
                "Model Information",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Model")
        st.write(model_path.as_posix() if model_path else "Not loaded")
        return page


def main():
    inject_css()
    model, model_path, model_error = load_model()
    data, data_path, data_error = load_dataset()
    page = sidebar_menu(model_path)

    if model_error:
        st.error(f"Model loading problem: {model_error}")
    if data_error:
        st.info(f"Dataset note: {data_error}")

    if page == "Home":
        home_page(model_path, data, data_path)
    elif page == "Single Customer Prediction":
        if model is None:
            st.error("Prediction is unavailable until a saved model is available.")
        else:
            single_customer_prediction_page(model)
    elif page == "Batch CSV Prediction":
        if model is None:
            st.error("Batch prediction is unavailable until a saved model is available.")
        else:
            batch_csv_prediction_page(model)
    elif page == "Prediction History":
        prediction_history_page()
    elif page == "Exploratory Data Analysis":
        eda_page(data)
    elif page == "Model Performance":
        model_performance_page()
    elif page == "Model Information":
        model_information_page(model_path)


if __name__ == "__main__":
    main()
