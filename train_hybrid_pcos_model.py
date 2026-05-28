import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


DATA_PATH = "PCOS_data.csv"
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

TARGET_COL = "PCOS (Y/N)"
ID_COLS = {"Sl. No", "Patient File No."}


def normalize_col(col: str) -> str:
    return "".join(ch.lower() for ch in str(col) if ch.isalnum())


def find_col(df: pd.DataFrame, candidates):
    lookup = {normalize_col(c): c for c in df.columns}
    for name in candidates:
        key = normalize_col(name)
        if key in lookup:
            return lookup[key]
    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce",
    )


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Convert obvious numeric/object columns safely.
    for col in df.columns:
        if col not in [TARGET_COL]:
            df[col] = coerce_numeric(df[col])

    # Target
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("Int64")

    # Derived features
    fsh_col = find_col(df, ["FSH(mIU/mL)"])
    lh_col = find_col(df, ["LH(mIU/mL)"])
    follicle_l_col = find_col(df, ["Follicle No. (L)"])
    follicle_r_col = find_col(df, ["Follicle No. (R)"])

    if fsh_col and lh_col:
        df["LH/FSH Ratio"] = df[lh_col] / df[fsh_col].replace({0: np.nan})
        df["FSH/LH Ratio"] = df[fsh_col] / df[lh_col].replace({0: np.nan})

    if follicle_l_col and follicle_r_col:
        df["Follicle Count"] = df[follicle_l_col].fillna(0) + df[follicle_r_col].fillna(0)

    # Keep only rows with target
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Remove IDs if present
    for c in list(df.columns):
        if c in ID_COLS:
            df.drop(columns=c, inplace=True)

    # Exclude target from feature list
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    # Impute numerics with median
    medians = {}
    for c in feature_cols:
        med = float(df[c].median()) if df[c].notna().any() else 0.0
        medians[c] = med
        df[c] = df[c].fillna(med)

    # Some features are naturally binary/ordinal; keep them as numbers
    return df, feature_cols, medians


def train_model():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    prepared, feature_cols, medians = prepare_dataframe(df)

    X = prepared[feature_cols].copy()
    y = prepared[TARGET_COL].copy()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model = VotingClassifier(
        estimators=[("random_forest", rf), ("xgboost", xgb)],
        voting="soft",
        weights=[1, 1],
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }

    # Ensemble feature importance approximation: average normalized importance
    rf_fitted = model.named_estimators_["random_forest"]
    xgb_fitted = model.named_estimators_["xgboost"]
    rf_imp = getattr(rf_fitted, "feature_importances_", np.zeros(len(feature_cols)))
    xgb_imp = getattr(xgb_fitted, "feature_importances_", np.zeros(len(feature_cols)))
    rf_imp = rf_imp / (rf_imp.sum() + 1e-9)
    xgb_imp = xgb_imp / (xgb_imp.sum() + 1e-9)
    avg_imp = (rf_imp + xgb_imp) / 2.0
    feature_importance = dict(sorted(zip(feature_cols, avg_imp.tolist()), key=lambda x: x[1], reverse=True))

    # Save artifacts
    joblib.dump(model, ARTIFACT_DIR / "pcos_hybrid_model.joblib")
    joblib.dump(feature_cols, ARTIFACT_DIR / "feature_cols.joblib")
    joblib.dump(medians, ARTIFACT_DIR / "medians.joblib")
    joblib.dump(feature_importance, ARTIFACT_DIR / "feature_importance.joblib")
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== TRAINING COMPLETE ===")
    print(f"Rows used: {len(prepared)}")
    print(f"Features used: {len(feature_cols)}")
    print(f"Train size: {metrics['n_train']} | Test size: {metrics['n_test']}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print("Confusion Matrix:", metrics["confusion_matrix"])
    print(f"\nSaved artifacts in: {ARTIFACT_DIR.resolve()}")


if __name__ == "__main__":
    train_model()
