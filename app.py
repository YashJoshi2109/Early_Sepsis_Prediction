import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, Any
import pickle
import os

# -----------------------------
# Global settings / constants
# -----------------------------

RANDOM_STATE = 0

# Columns you dropped in the notebook
COLUMNS_DROP = {
    "Unnamed: 0",
    "SBP",
    "DBP",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "pH",
    "PaCO2",
    "Alkalinephos",
    "Calcium",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "PTT",
    "Fibrinogen",
    "Unit1",
    "Unit2",
}

# High-null or non-feature columns (including Patient_ID)
HIGH_NULL_COLS = [
    "TroponinI",
    "Bilirubin_direct",
    "AST",
    "Bilirubin_total",
    "Lactate",
    "SaO2",
    "FiO2",
    "Unit",
    "Patient_ID",
]

# Columns you log-transformed in the notebook
LOG_COLS = ["MAP", "BUN", "Creatinine", "Glucose", "WBC", "Platelets"]

# Columns you standard-scaled
SCALE_COLS = [
    "HR",
    "O2Sat",
    "Temp",
    "MAP",
    "Resp",
    "BUN",
    "Chloride",
    "Creatinine",
    "Glucose",
    "Hct",
    "Hgb",
    "WBC",
    "Platelets",
]

TARGET_COL = "SepsisLabel"


# -----------------------------
# Data loading / preprocessing
# -----------------------------

@st.cache_data(show_spinner=False)
def load_combined_dataset(path: str = "Dataset.csv") -> pd.DataFrame:
    """Load the full combined dataset (both hospitals)."""
    df = pd.read_csv(path)
    return df


def split_train_test_hospitals(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recreate your split:
    - Training: Patient_IDs whose length != 6
    - Test (other hospital): Patient_IDs whose length == 6
    """
    rows_to_drop_train = df.loc[df["Patient_ID"].apply(lambda x: len(str(x)) == 6)]
    df_train = df.drop(rows_to_drop_train.index)

    rows_to_drop_test = df.loc[df["Patient_ID"].apply(lambda x: len(str(x)) != 6)]
    df_test = df.drop(rows_to_drop_test.index)

    return df_train, df_test


def preprocess_frame(
    df: pd.DataFrame,
    fit_scaler: bool = True,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply the same preprocessing steps you used in the notebook:
    - Combine Unit1 + Unit2 → Unit
    - Drop redundant columns
    - Group by Patient_ID → bfill + ffill
    - Drop high-null columns + Patient_ID
    - Log transform selected columns
    - Standard scale vitals/labs
    - One-hot encode Gender
    - Drop remaining NaNs
    """
    df = df.copy()

    # Combine Unit1 + Unit2
    if "Unit1" in df.columns and "Unit2" in df.columns:
        df = df.assign(Unit=df["Unit1"] + df["Unit2"])

    # Drop redundant columns
    drop_cols = [c for c in COLUMNS_DROP if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Group by Patient_ID and bfill/ffill if Patient_ID is present
    if "Patient_ID" in df.columns:
        grouped_by_patient = df.groupby("Patient_ID")
        df = grouped_by_patient.apply(lambda x: x.bfill().ffill())
        # After groupby-apply, bring Patient_ID back as a column index (not level)
        df = df.reset_index(level=0, drop=True)

    # Drop high-null / non-feature columns
    drop_cols_high = [c for c in HIGH_NULL_COLS if c in df.columns]
    if drop_cols_high:
        df = df.drop(columns=drop_cols_high)

    # Log transform selected columns
    for c in LOG_COLS:
        if c in df.columns:
            df[c] = np.log(df[c] + 1)

    # Standard scaling
    numeric_to_scale = [c for c in SCALE_COLS if c in df.columns]
    if numeric_to_scale:
        if fit_scaler or scaler is None:
            scaler = StandardScaler()
            df[numeric_to_scale] = scaler.fit_transform(df[numeric_to_scale])
        else:
            df[numeric_to_scale] = scaler.transform(df[numeric_to_scale])

    # One-hot encode gender
    if "Gender" in df.columns:
        one_hot = pd.get_dummies(df["Gender"], prefix="Gender")
        df = df.join(one_hot)
        df = df.drop(columns=["Gender"])

    # Final cleanup
    df = df.dropna()
    df.columns = df.columns.astype(str)

    return df, scaler


# -----------------------------
# Model training
# -----------------------------

@st.cache_resource(show_spinner=True)
def train_model() -> Dict:
    """
    Train the Random Forest model similarly to your notebook:
    - Train on df_train (hospital 1)
    - Preprocess (imputation, log, scaling, encoding)
    - Undersample majority class to 2:1 ratio
    - Train RF with n_estimators=300
    """
    combined = load_combined_dataset()
    df_train, _ = split_train_test_hospitals(combined)

    # Preprocess training frame
    df_train_proc, scaler = preprocess_frame(df_train, fit_scaler=True, scaler=None)

    if TARGET_COL not in df_train_proc.columns:
        raise ValueError(f"Target column '{TARGET_COL}' missing after preprocessing.")

    # Undersampling majority class to 2:1 ratio (as in your notebook)
    majority = df_train_proc[df_train_proc[TARGET_COL] == 0]
    minority = df_train_proc[df_train_proc[TARGET_COL] == 1]
    if len(minority) == 0:
        raise ValueError("No positive (sepsis) cases found in training data.")

    majority_subset = majority.sample(
        n=min(2 * len(minority), len(majority)),
        random_state=RANDOM_STATE,
    )
    df_balanced = pd.concat([majority_subset, minority])

    X = df_balanced.drop(columns=[TARGET_COL])
    y = df_balanced[TARGET_COL]

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Random Forest model (best config from notebook)
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # Validation metrics
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred, zero_division=0),
        "Recall": recall_score(y_val, y_pred, zero_division=0),
        "F1-score": f1_score(y_val, y_pred, zero_division=0),
        "AUC-ROC": roc_auc_score(y_val, y_proba),
        "ConfusionMatrix": confusion_matrix(y_val, y_pred),
    }

    return {
        "model": model,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "metrics": metrics,
    }


# -----------------------------
# Pretrained artifacts loader
# -----------------------------

def load_artifacts_from_pickle(obj: Any) -> Dict:
    """Load model/scaler/features from a pickle object.

    The pickle may contain:
    - a dict with keys: model, scaler, feature_names[, metrics]
    - a single sklearn model (fallback: caller must supply scaler/features)
    """
    try:
        loaded = pickle.load(obj)
    except Exception as e:
        raise ValueError(f"Failed to load pickle: {e}")

    # If it's a dict-like artifact bundle
    if isinstance(loaded, dict):
        model = loaded.get("model")
        scaler = loaded.get("scaler")
        feature_names = loaded.get("feature_names")
        metrics = loaded.get("metrics")
        if model is None or feature_names is None:
            raise ValueError(
                "Pickle dict must contain at least 'model' and 'feature_names'."
            )
        return {
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "metrics": metrics,
        }

    # Otherwise assume it's a bare model
    return {
        "model": loaded,
        "scaler": None,
        "feature_names": None,
        "metrics": None,
    }


# -----------------------------
# Visualization helpers
# -----------------------------

def plot_confusion_matrix(cm: np.ndarray) -> None:
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Sepsis", "Sepsis"],
        yticklabels=["No Sepsis", "Sepsis"],
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    st.pyplot(fig)


# -----------------------------
# Streamlit UI
# -----------------------------

def main():
    st.set_page_config(
        page_title="Early Sepsis Prediction",
        layout="wide",
        page_icon="🩺",
    )

    st.title("🩺 Early Sepsis Prediction Dashboard")
    st.write(
        "Interactively explore the sepsis prediction model, inspect its performance, "
        "and run predictions for new patient data."
    )

    with st.sidebar:
        st.header("Configuration")
        st.markdown(
            "This app uses the Random Forest model you built in the notebook "
            "on top of the imputed and normalized training data."
        )

    # Option to use a pre-trained model
    st.sidebar.subheader("Pre-trained model")
    use_pretrained = st.sidebar.checkbox("Use pre-trained .pkl model", value=False)
    uploaded_model = None
    artifacts: Dict

    if use_pretrained:
        mode = st.sidebar.radio("Source", ["Load from path", "Upload file"], horizontal=True)

        if mode == "Load from path":
            model_path = st.sidebar.text_input(
                "Local .pkl path (absolute or relative)", value="random_model_artifacts.pkl"
            )
            if st.sidebar.button("Load model from path"):
                if not model_path:
                    st.sidebar.error("Please provide a .pkl path.")
                    st.stop()
                if not os.path.exists(model_path):
                    st.sidebar.error(f"File not found: {model_path}")
                    st.stop()
                try:
                    with open(model_path, "rb") as f:
                        artifacts = load_artifacts_from_pickle(f)
                    st.sidebar.success(f"Loaded pre-trained model from {model_path}.")
                except Exception as e:
                    st.sidebar.error(f"Failed to load model: {e}")
                    st.stop()
            else:
                st.info("Provide the path and click 'Load model from path', or switch to Upload.")
                st.stop()
        else:
            uploaded_model = st.sidebar.file_uploader(
                "Upload model .pkl (dict with model, scaler, feature_names preferred)",
                type=["pkl", "pickle"],
                key="model_pkl",
            )
            if uploaded_model is not None:
                try:
                    artifacts = load_artifacts_from_pickle(uploaded_model)
                    st.sidebar.success("Loaded pre-trained model.")
                except Exception as e:
                    st.sidebar.error(f"Failed to load model: {e}")
                    st.stop()
            else:
                st.info("Please upload a .pkl file, or switch to 'Load from path'.")
                st.stop()
    else:
        with st.spinner("Training / loading model..."):
            artifacts = train_model()

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    feature_names = artifacts["feature_names"]
    metrics = artifacts.get("metrics")

    tab_overview, tab_batch, tab_manual = st.tabs(
        ["📊 Model overview", "📂 Batch prediction (CSV)", "🧍 Manual patient input"]
    )

    # -------- Model Overview Tab --------
    with tab_overview:
        st.subheader("Validation metrics")
        if metrics is not None:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
            c2.metric("Precision", f"{metrics['Precision']:.3f}")
            c3.metric("Recall", f"{metrics['Recall']:.3f}")
            c4.metric("F1-score", f"{metrics['F1-score']:.3f}")
            c5.metric("AUC-ROC", f"{metrics['AUC-ROC']:.3f}")

            st.markdown("#### Confusion matrix")
            plot_confusion_matrix(metrics["ConfusionMatrix"]) 
        else:
            st.info("Metrics unavailable for uploaded pre-trained model.")

        st.markdown("#### Feature list used by the model")
        st.write(feature_names)

    # -------- Batch Prediction Tab --------
    with tab_batch:
        st.subheader("Batch prediction from CSV")
        st.markdown(
            "Upload a CSV file with the same structure as the original `Dataset.csv` "
            "or `data_part2.csv` (one row per ICU hour per patient)."
        )
        uploaded = st.file_uploader(
            "Upload CSV for prediction", type=["csv"], key="batch_uploader"
        )

        if uploaded is not None:
            try:
                df_new = pd.read_csv(uploaded)
                st.write("Raw shape:", df_new.shape)
                st.write(df_new.head())

                df_proc, _ = preprocess_frame(df_new, fit_scaler=False, scaler=scaler)

                missing = [c for c in feature_names if c not in df_proc.columns]
                if missing:
                    st.error(
                        "The uploaded data is missing some features expected by the "
                        f"model:\n\n{missing}"
                    )
                else:
                    X_new = df_proc[feature_names]
                    probs = model.predict_proba(X_new)[:, 1]
                    preds = (probs >= 0.5).astype(int)

                    result_df = df_proc.copy()
                    result_df["Sepsis_Prob"] = probs
                    result_df["Sepsis_Pred"] = preds

                    st.markdown("#### Predictions (first few rows)")
                    st.write(result_df.head())

                    st.download_button(
                        "Download predictions as CSV",
                        data=result_df.to_csv(index=False),
                        file_name="sepsis_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Failed to process uploaded file: {e}")

    # -------- Manual Input Tab --------
    with tab_manual:
        st.subheader("Manual patient input")
        st.markdown(
            "Enter a single time-point of patient vitals and labs. "
            "Only a subset of key features is exposed here; the rest will be "
            "filled with typical values."
        )

        typical = {
            "HR": 90.0,
            "O2Sat": 97.0,
            "Temp": 37.0,
            "MAP": 80.0,
            "Resp": 20.0,
            "BUN": 18.0,
            "Chloride": 103.0,
            "Creatinine": 1.0,
            "Glucose": 110.0,
            "Hct": 38.0,
            "Hgb": 12.0,
            "WBC": 9.0,
            "Platelets": 250.0,
            "Age": 60.0,
            "HospAdmTime": 0.0,
            "ICULOS": 6.0,
        }

        col1, col2, col3 = st.columns(3)
        with col1:
            hr = st.slider("Heart rate (HR)", 30, 200, int(typical["HR"]))
            o2 = st.slider("O₂ saturation (O2Sat)", 50, 100, int(typical["O2Sat"]))
            temp = st.slider("Temperature (°C)", 30.0, 42.0, float(typical["Temp"]))
            resp = st.slider("Respiratory rate", 5, 50, int(typical["Resp"]))
        with col2:
            map_ = st.slider("Mean arterial pressure (MAP)", 30, 150, int(typical["MAP"]))
            bun = st.slider("BUN", 1, 80, int(typical["BUN"]))
            creatinine = st.slider("Creatinine", 0.0, 10.0, float(typical["Creatinine"]))
            glucose = st.slider("Glucose", 40, 500, int(typical["Glucose"]))
        with col3:
            wbc = st.slider("WBC", 1, 50, int(typical["WBC"]))
            platelets = st.slider("Platelets", 10, 800, int(typical["Platelets"]))
            hct = st.slider("Hematocrit (Hct)", 10, 60, int(typical["Hct"]))
            hgb = st.slider("Hemoglobin (Hgb)", 5, 20, int(typical["Hgb"]))

        gender = st.radio("Gender", options=["M", "F"], horizontal=True)
        age = st.slider("Age", 18, 100, int(typical["Age"]))
        hosp_adm = st.number_input(
            "HospAdmTime",
            value=float(typical["HospAdmTime"]),
            help="Time (in whatever units your dataset uses) from hospital admission.",
        )
        iculos = st.number_input(
            "ICULOS (hours since ICU admit)",
            value=float(typical["ICULOS"]),
        )

        if st.button("Predict sepsis risk"):
            # Build a dummy raw row (like original dataset) to pass through preprocess_frame
            dummy = pd.DataFrame(
                [
                    {
                        "HR": hr,
                        "O2Sat": o2,
                        "Temp": temp,
                        "MAP": map_,
                        "Resp": resp,
                        "BUN": bun,
                        "Chloride": typical["Chloride"],
                        "Creatinine": creatinine,
                        "Glucose": glucose,
                        "Hct": hct,
                        "Hgb": hgb,
                        "WBC": wbc,
                        "Platelets": platelets,
                        "Age": age,
                        "Gender": gender,
                        "HospAdmTime": hosp_adm,
                        "ICULOS": iculos,
                        "Unit1": 0,
                        "Unit2": 1,
                        "SepsisLabel": 0,
                        "Patient_ID": 0,
                    }
                ]
            )

            # Use the same preprocessing (with the already-fitted scaler)
            dummy_proc, _ = preprocess_frame(dummy, fit_scaler=False, scaler=scaler)

            # Align with training feature set
            row = {name: 0.0 for name in feature_names}
            for col in feature_names:
                if col in dummy_proc.columns:
                    row[col] = float(dummy_proc[col].iloc[0])

            df_row = pd.DataFrame([row], columns=feature_names)

            proba = model.predict_proba(df_row)[:, 1][0]
            pred = int(proba >= 0.3)

            st.markdown("### Prediction result")
            st.write(f"**Sepsis probability:** `{proba:.3f}`")
            st.write("**Predicted label:**", "🟥 Sepsis" if pred == 1 else "🟩 No sepsis")


if __name__ == "__main__":
    main()
