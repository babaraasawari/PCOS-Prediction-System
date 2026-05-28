from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "pcos_hybrid_model.joblib"
FEATURES_PATH = ARTIFACT_DIR / "feature_cols.joblib"
MEDIANS_PATH = ARTIFACT_DIR / "medians.joblib"
IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
DATA_PATH = "PCOS_data.csv"

DERIVED_FEATURES = {"LH/FSH Ratio", "FSH/LH Ratio", "Follicle Count"}

st.set_page_config(
    page_title="PCOS Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #f7fbff 0%, #eef5ff 100%);
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f4c81 0%, #1d6fb8 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(15,76,129,0.18);
        margin-bottom: 1rem;
    }
    .hero h1 { color: white; margin-bottom: 0.2rem; }
    .hero p { color: #e9f4ff; margin-top: 0.1rem; font-size: 1rem; }
    .card {
        background: white;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 22px rgba(27, 59, 94, 0.08);
        border: 1px solid rgba(30, 78, 120, 0.08);
    }
    .small-note {
        color: #52616b;
        font-size: 0.92rem;
    }
    .warning-box {
        background: #fff8e6;
        border-left: 5px solid #f5b301;
        padding: 0.9rem 1rem;
        border-radius: 12px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def normalize_col(col: str) -> str:
    return "".join(ch.lower() for ch in str(col) if ch.isalnum())

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURES_PATH)
    medians = joblib.load(MEDIANS_PATH)
    importance = joblib.load(IMPORTANCE_PATH)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return model, feature_cols, medians, importance, metrics

def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    return df

def fmt_feature(name: str) -> str:
    return name.replace("  ", " ").strip()

def get_feature_group(feature_name: str):
    if feature_name in DERIVED_FEATURES:
        return "Derived Features"
    n = normalize_col(feature_name)
    if any(k in n for k in ["age", "weight", "height", "bmi", "bloodgroup", "marraigestatus"]):
        return "Demographics"
    if any(k in n for k in ["pulse", "rr", "hb", "bpsystolic", "bpdiastolic", "rbs"]):
        return "Vitals"
    if any(k in n for k in ["cycle", "pregnant", "abort", "betahcg"]):
        return "Cycle & Reproductive History"
    if any(k in n for k in ["fsh", "lh", "tsh", "amh", "prl", "vitd3", "prg", "insulin", "testosterone"]):
        return "Hormonal & Metabolic"
    if any(k in n for k in ["weightgain", "hairgrowth", "skindarkening", "hairloss", "pimples", "fastfood", "regexercise"]):
        return "Symptoms & Lifestyle"
    if any(k in n for k in ["hip", "waist", "waisthipratio", "follicle", "avgfsize", "endometrium"]):
        return "Ultrasound & Body Measurements"
    return "Other"

def num_input(label, default, min_value=None, max_value=None, step=None):
    kwargs = {"value": float(default)}
    if min_value is not None:
        kwargs["min_value"] = float(min_value)
    if max_value is not None:
        kwargs["max_value"] = float(max_value)
    if step is not None:
        kwargs["step"] = float(step)
    return st.number_input(label, **kwargs)

def main():
    st.markdown(
        """
        <div class="hero">
            <h1>🩺 PCOS Prediction System</h1>
            <p>Hybrid ensemble model using Random Forest + XGBoost for explainable PCOS prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, feature_cols, medians, importance, metrics = load_artifacts()
    df = load_dataset()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{metrics['precision']:.3f}")
    col3.metric("Recall", f"{metrics['recall']:.3f}")
    col4.metric("F1-score", f"{metrics['f1']:.3f}")

    st.markdown(
        """
        <div class="warning-box">
            <b>Important:</b> The supplied CSV does not include <b>Insulin level</b> or <b>Testosterone level</b>.
            Those fields are shown below as <b>future-extension inputs</b>, but they are not used in the current trained model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Patient Form", "Prediction", "Model Performance", "Feature Importance", "Dataset Notes"]
    )

    with tab1:
        st.subheader("Enter patient details")
        st.caption("The form is grouped to stay clean and presentation-friendly.")

        with st.form("pcos_form"):
            input_values = {}

            sections = [
                ("Demographics", [c for c in feature_cols if get_feature_group(c) == "Demographics"]),
                ("Symptoms & Lifestyle", [c for c in feature_cols if get_feature_group(c) == "Symptoms & Lifestyle"]),
                ("Cycle & Reproductive History", [c for c in feature_cols if get_feature_group(c) == "Cycle & Reproductive History"]),
                ("Hormonal & Metabolic", [c for c in feature_cols if get_feature_group(c) == "Hormonal & Metabolic"]),
                ("Ultrasound & Body Measurements", [c for c in feature_cols if get_feature_group(c) == "Ultrasound & Body Measurements"]),
            ]

            # Two-column visual layout
            left_col, right_col = st.columns(2)

            # Helper to draw a feature form control
            def render_feature(feat: str):
                if feat in DERIVED_FEATURES:
                    return
                default = medians.get(feat, 0.0)
                n = normalize_col(feat)

                if n in {"weightgainedyn", "hairgrowthyn", "skindarkeningyn", "hairlossyn", "pimplesyn", "fastfoodyn", "regexerciseyn"}:
                    input_values[feat] = st.selectbox(feat, ["No", "Yes"], index=1 if default >= 0.5 else 0)
                elif n == "cycleri":
                    input_values[feat] = st.selectbox(feat, ["Regular", "Irregular"], index=0 if default >= 0.5 else 1)
                elif n == "bloodgroup":
                    input_values[feat] = st.selectbox(feat, [1, 2, 3, 4, 5, 6, 7, 8], index=max(0, min(7, int(round(default)) - 1)))
                elif n in {"ageyrs", "cyclelengthdays", "pulseratebpm", "rrbreathsmin", "hipinch", "waistinch", "folliclenol", "folliclenor", "bpsystolicmmhg", "bpdiastolicmmhg"}:
                    input_values[feat] = num_input(feat, default, min_value=0, max_value=250, step=1)
                else:
                    input_values[feat] = num_input(feat, default, min_value=0, step=0.1)

            with left_col:
                for title, feats in sections[:3]:
                    st.markdown(f"#### {title}")
                    for feat in feats:
                        render_feature(feat)

            with right_col:
                for title, feats in sections[3:]:
                    st.markdown(f"#### {title}")
                    for feat in feats:
                        render_feature(feat)

                st.markdown("#### Derived Features")
                st.info("LH/FSH Ratio and Follicle Count are calculated automatically from the inputs.")

                if "LH(mIU/mL)" in feature_cols and "FSH(mIU/mL)" in feature_cols:
                    lh = float(input_values.get("LH(mIU/mL)", medians.get("LH(mIU/mL)", 0)))
                    fsh = float(input_values.get("FSH(mIU/mL)", medians.get("FSH(mIU/mL)", 1)))
                    input_values["LH/FSH Ratio"] = lh / fsh if fsh else 0.0
                    input_values["FSH/LH Ratio"] = fsh / lh if lh else 0.0

                if "Follicle No. (L)" in feature_cols and "Follicle No. (R)" in feature_cols:
                    fl = float(input_values.get("Follicle No. (L)", medians.get("Follicle No. (L)", 0)))
                    fr = float(input_values.get("Follicle No. (R)", medians.get("Follicle No. (R)", 0)))
                    input_values["Follicle Count"] = fl + fr

                st.markdown("#### Additional Report Inputs (future extension)")
                st.number_input("Insulin level", min_value=0.0, value=12.0, step=0.1)
                st.number_input("Testosterone level", min_value=0.0, value=0.6, step=0.01)
                st.caption("These are captured for future expansion. They are not used in the current model because the supplied CSV does not contain them.")

            submitted = st.form_submit_button("Predict PCOS")

    with tab2:
        st.subheader("Prediction result")
        st.write("Submit the patient form to generate the outcome and probability score.")

        if submitted:
            row = {}
            for feat in feature_cols:
                val = input_values.get(feat, medians.get(feat, 0))
                if isinstance(val, str):
                    if val == "Yes":
                        row[feat] = 1
                    elif val == "No":
                        row[feat] = 0
                    elif val == "Regular":
                        row[feat] = 1
                    elif val == "Irregular":
                        row[feat] = 0
                    else:
                        row[feat] = 0
                else:
                    row[feat] = float(val)

            # Recompute derived features for safety
            if "LH/FSH Ratio" in feature_cols and "LH(mIU/mL)" in feature_cols and "FSH(mIU/mL)" in feature_cols:
                lh = float(row.get("LH(mIU/mL)", 0))
                fsh = float(row.get("FSH(mIU/mL)", 0))
                row["LH/FSH Ratio"] = lh / fsh if fsh else 0.0
            if "FSH/LH Ratio" in feature_cols and "LH(mIU/mL)" in feature_cols and "FSH(mIU/mL)" in feature_cols:
                lh = float(row.get("LH(mIU/mL)", 0))
                fsh = float(row.get("FSH(mIU/mL)", 0))
                row["FSH/LH Ratio"] = fsh / lh if lh else 0.0
            if "Follicle Count" in feature_cols:
                row["Follicle Count"] = float(row.get("Follicle No. (L)", 0)) + float(row.get("Follicle No. (R)", 0))

            X_in = pd.DataFrame([row], columns=feature_cols)
            X_in = X_in.fillna(pd.Series(medians))
            pred = int(model.predict(X_in)[0])
            prob = float(model.predict_proba(X_in)[0][1])

            a, b = st.columns([1, 1])
            with a:
                if pred == 1:
                    st.error("PCOS Prediction: YES")
                else:
                    st.success("PCOS Prediction: NO")
                st.metric("PCOS probability", f"{prob*100:.2f}%")

            with b:
                st.markdown(
                    """
                    <div class="card">
                        <b>Interpretation</b><br><br>
                        The hybrid ensemble combines Random Forest and XGBoost. The final prediction is based on the supplied clinical and lifestyle inputs.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if "LH/FSH Ratio" in row:
                    st.write(f"**LH/FSH Ratio:** {row['LH/FSH Ratio']:.3f}")
                if "Follicle Count" in row:
                    st.write(f"**Follicle Count:** {row['Follicle Count']:.0f}")

    with tab3:
        st.subheader("Model performance")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Metrics")
            st.dataframe(
                pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
                    "Value": [
                        metrics["accuracy"],
                        metrics["precision"],
                        metrics["recall"],
                        metrics["f1"],
                        metrics["roc_auc"],
                    ],
                }),
                use_container_width=True,
                hide_index=True,
            )
        with c2:
            st.markdown("#### Confusion matrix")
            cm = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig, clear_figure=True)

    with tab4:
        st.subheader("Feature importance")
        top_n = 12
        imp_items = list(importance.items())[:top_n]
        names = [k for k, _ in imp_items]
        vals = [v for _, v in imp_items]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=vals, y=names, ax=ax)
        ax.set_title("Top Feature Importance (Random Forest + XGBoost)")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        st.pyplot(fig, clear_figure=True)

    with tab5:
        st.subheader("Dataset notes")
        st.write(f"Rows in supplied dataset: **{len(df)}**")
        st.write(f"Features used by the current trained model: **{len(feature_cols)}**")
        st.write("Additional fields requested by you:")
        st.markdown(
            """
            - LH/FSH Ratio: derived and used in the model
            - Insulin level: not present in the provided CSV
            - Testosterone level: not present in the provided CSV
            - Follicle count: derived and used in the model
            """
        )
        st.caption("If you later add a richer dataset, the same structure can be extended to include insulin and testosterone directly.")

if __name__ == "__main__":
    main()
