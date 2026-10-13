"""
Phase 2: Rigorous Model Benchmarking, Selection, and Threshold Analysis
Customer Churn Prediction & Retention Intelligence System

This script performs:
1. Baseline reproduction (Logistic Regression & Random Forest holdout metrics).
2. 5-Fold Stratified Cross-Validation on training data across:
   - Logistic Regression (unweighted & balanced)
   - Random Forest (unweighted & balanced)
   - XGBoost (unweighted & tuned/weighted)
   - LightGBM (unweighted & tuned/weighted)
   - CatBoost (unweighted & tuned/weighted)
3. Leakage-safe preprocessing using ColumnTransformer & OneHotEncoder.
4. Out-of-fold (OOF) threshold optimization for Recall and F1-score.
5. Probability calibration quality (Brier score & calibration curve).
6. Single final evaluation of the selected model on the untouched holdout test set.
7. Preservation of all original production artifacts.
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# Define directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
ORIGINAL_MODEL_DIR = PROJECT_ROOT / "model"
EXP_DIR = ORIGINAL_MODEL_DIR / "experiments"
EXP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PHASE 2: RIGOROUS MODEL BENCHMARKING & SELECTION")
print(f"Project root: {PROJECT_ROOT}")
print(f"Output experiment directory: {EXP_DIR}")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. LOAD & PARTITION DATA (Replicating exact train/test split)
# ---------------------------------------------------------------------------
print("\n[Step 1] Loading raw dataset and replicating train/test partition...")
df_raw = pd.read_csv(DATA_PATH)
df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")
df_clean = df_raw.dropna(subset=["TotalCharges"]).copy()
df_clean = df_clean.drop(columns=["customerID"])
df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})

X = df_clean.drop(columns=["Churn"])
y = df_clean["Churn"]

# Explicitly categorize feature columns
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
# SeniorCitizen is int, explicitly treated as binary categorical
categorical_features = [
    col for col in X.columns if col not in numeric_features
]

print(f"Dataset: {len(df_clean)} records. Churn positive class rate: {y.mean():.4f}")
print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# Split identically to preprocess.py
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train partition: {len(X_train_raw)} rows (Churn: {y_train.sum()}, {y_train.mean():.4f})")
print(f"Test partition (Untouched holdout): {len(X_test_raw)} rows (Churn: {y_test.sum()}, {y_test.mean():.4f})")

# ---------------------------------------------------------------------------
# 2. BASELINE REPRODUCTION (Holdout Test Set Evaluation)
# ---------------------------------------------------------------------------
print("\n[Step 2] Establishing and recording existing baseline on test set...")
saved_lr_model = joblib.load(ORIGINAL_MODEL_DIR / "churn_model.pkl")
saved_X_test = joblib.load(ORIGINAL_MODEL_DIR / "X_test.pkl")
saved_y_test = joblib.load(ORIGINAL_MODEL_DIR / "y_test.pkl")

# Baseline Logistic Regression
lr_pred = saved_lr_model.predict(saved_X_test)
lr_prob = saved_lr_model.predict_proba(saved_X_test)[:, 1]
lr_cm = confusion_matrix(saved_y_test, lr_pred)

# Baseline Random Forest (from train.py specification)
saved_X_train = joblib.load(ORIGINAL_MODEL_DIR / "X_train.pkl")
saved_y_train = joblib.load(ORIGINAL_MODEL_DIR / "y_train.pkl")
baseline_rf = RandomForestClassifier(n_estimators=200, random_state=42)
baseline_rf.fit(saved_X_train, saved_y_train)
rf_pred = baseline_rf.predict(saved_X_test)
rf_prob = baseline_rf.predict_proba(saved_X_test)[:, 1]
rf_cm = confusion_matrix(saved_y_test, rf_pred)

baseline_results = {
    "Logistic Regression (Production Baseline)": {
        "threshold": 0.50,
        "accuracy": float(accuracy_score(saved_y_test, lr_pred)),
        "precision": float(precision_score(saved_y_test, lr_pred)),
        "recall": float(recall_score(saved_y_test, lr_pred)),
        "f1_score": float(f1_score(saved_y_test, lr_pred)),
        "roc_auc": float(roc_auc_score(saved_y_test, lr_prob)),
        "pr_auc": float(average_precision_score(saved_y_test, lr_prob)),
        "confusion_matrix": lr_cm.tolist(),
        "tn": int(lr_cm[0, 0]),
        "fp": int(lr_cm[0, 1]),
        "fn": int(lr_cm[1, 0]),
        "tp": int(lr_cm[1, 1]),
    },
    "Random Forest Classifier (Benchmark Baseline)": {
        "threshold": 0.50,
        "accuracy": float(accuracy_score(saved_y_test, rf_pred)),
        "precision": float(precision_score(saved_y_test, rf_pred)),
        "recall": float(recall_score(saved_y_test, rf_pred)),
        "f1_score": float(f1_score(saved_y_test, rf_pred)),
        "roc_auc": float(roc_auc_score(saved_y_test, rf_prob)),
        "pr_auc": float(average_precision_score(saved_y_test, rf_prob)),
        "confusion_matrix": rf_cm.tolist(),
        "tn": int(rf_cm[0, 0]),
        "fp": int(rf_cm[0, 1]),
        "fn": int(rf_cm[1, 0]),
        "tp": int(rf_cm[1, 1]),
    },
}

with open(EXP_DIR / "baseline_results.json", "w") as f:
    json.dump(baseline_results, f, indent=2)

print("Baseline Results recorded:")
for m_name, m_res in baseline_results.items():
    print(f"  {m_name}: Acc={m_res['accuracy']:.4f}, Prec={m_res['precision']:.4f}, Rec={m_res['recall']:.4f}, F1={m_res['f1_score']:.4f}, ROC-AUC={m_res['roc_auc']:.4f}, PR-AUC={m_res['pr_auc']:.4f}")

# ---------------------------------------------------------------------------
# 3. FAIR LEAKAGE-SAFE PREPROCESSING PIPELINES
# ---------------------------------------------------------------------------
print("\n[Step 3] Building leakage-safe preprocessing pipelines...")

# Scaled pipeline for linear models (Logistic Regression)
preprocessor_scaled = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical_features),
    ]
)

# Unscaled pipeline for tree-based models (trees do not require standard scaling)
preprocessor_trees = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical_features),
    ]
)

scale_pos_weight_val = (len(y_train) - y_train.sum()) / y_train.sum()  # ~2.76
print(f"Calculated scale_pos_weight for positive class: {scale_pos_weight_val:.4f}")

# Candidate models dictionary
candidates = {
    "Logistic Regression (Default)": (
        preprocessor_scaled,
        LogisticRegression(max_iter=1000, random_state=42),
    ),
    "Logistic Regression (Balanced)": (
        preprocessor_scaled,
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    ),
    "Logistic Regression (Tuned C=0.1, Balanced)": (
        preprocessor_scaled,
        LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42),
    ),
    "Random Forest (Default)": (
        preprocessor_trees,
        RandomForestClassifier(n_estimators=200, random_state=42),
    ),
    "Random Forest (Balanced, max_depth=8)": (
        preprocessor_trees,
        RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=8, min_samples_leaf=4, random_state=42),
    ),
    "XGBoost (Default)": (
        preprocessor_trees,
        xgb.XGBClassifier(eval_metric="logloss", random_state=42),
    ),
    "XGBoost (Weighted & Tuned)": (
        preprocessor_trees,
        xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight_val,
            max_depth=4,
            learning_rate=0.05,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        ),
    ),
    "LightGBM (Default)": (
        preprocessor_trees,
        lgb.LGBMClassifier(random_state=42, verbose=-1),
    ),
    "LightGBM (Balanced & Tuned)": (
        preprocessor_trees,
        lgb.LGBMClassifier(
            scale_pos_weight=scale_pos_weight_val,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.05,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        ),
    ),
    "CatBoost (Default)": (
        preprocessor_trees,
        CatBoostClassifier(iterations=200, learning_rate=0.05, random_seed=42, verbose=0),
    ),
    "CatBoost (Balanced & Tuned)": (
        preprocessor_trees,
        CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=4,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
        ),
    ),
}

# ---------------------------------------------------------------------------
# 4. STRATIFIED 5-FOLD CROSS-VALIDATION
# ---------------------------------------------------------------------------
print("\n[Step 4] Running Stratified 5-Fold Cross-Validation on Training Data...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results_summary = []
oof_predictions = {}

for model_name, (preproc, model_inst) in candidates.items():
    print(f"\nEvaluating: {model_name}...")
    pipeline = Pipeline([
        ("preprocessor", preproc),
        ("classifier", model_inst),
    ])

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    fold_rocs = []
    fold_prs = []

    # Array to store out-of-fold probability predictions
    oof_probs = np.zeros(len(X_train_raw))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_raw, y_train), 1):
        X_tr, X_val = X_train_raw.iloc[train_idx], X_train_raw.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.50).astype(int)

        oof_probs[val_idx] = val_probs

        fold_accuracies.append(accuracy_score(y_val, val_preds))
        fold_precisions.append(precision_score(y_val, val_preds, zero_division=0))
        fold_recalls.append(recall_score(y_val, val_preds))
        fold_f1s.append(f1_score(y_val, val_preds))
        fold_rocs.append(roc_auc_score(y_val, val_probs))
        fold_prs.append(average_precision_score(y_val, val_probs))

    oof_predictions[model_name] = oof_probs

    summary_entry = {
        "Model": model_name,
        "Recall_Mean": np.mean(fold_recalls),
        "Recall_Std": np.std(fold_recalls),
        "F1_Mean": np.mean(fold_f1s),
        "F1_Std": np.std(fold_f1s),
        "Precision_Mean": np.mean(fold_precisions),
        "Precision_Std": np.std(fold_precisions),
        "ROC_AUC_Mean": np.mean(fold_rocs),
        "ROC_AUC_Std": np.std(fold_rocs),
        "PR_AUC_Mean": np.mean(fold_prs),
        "PR_AUC_Std": np.std(fold_prs),
        "Accuracy_Mean": np.mean(fold_accuracies),
        "Accuracy_Std": np.std(fold_accuracies),
    }
    cv_results_summary.append(summary_entry)

    print(
        f"  Recall: {summary_entry['Recall_Mean']:.4f} ± {summary_entry['Recall_Std']:.4f} | "
        f"F1: {summary_entry['F1_Mean']:.4f} ± {summary_entry['F1_Std']:.4f} | "
        f"Prec: {summary_entry['Precision_Mean']:.4f} ± {summary_entry['Precision_Std']:.4f} | "
        f"ROC-AUC: {summary_entry['ROC_AUC_Mean']:.4f} ± {summary_entry['ROC_AUC_Std']:.4f} | "
        f"PR-AUC: {summary_entry['PR_AUC_Mean']:.4f} ± {summary_entry['PR_AUC_Std']:.4f}"
    )

cv_df = pd.DataFrame(cv_results_summary).sort_values(by="F1_Mean", ascending=False).reset_index(drop=True)
cv_df.to_csv(EXP_DIR / "cv_model_comparison.csv", index=False)
print("\n" + "=" * 90)
print("5-FOLD CV SUMMARY RANKED BY F1-SCORE (Prioritizing Recall & F1):")
print("=" * 90)
cols_display = ["Model", "Recall_Mean", "F1_Mean", "Precision_Mean", "ROC_AUC_Mean", "PR_AUC_Mean", "Accuracy_Mean"]
print(cv_df[cols_display].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------------------------------------------------------------------------
# 5. MODEL SELECTION & THRESHOLD OPTIMIZATION
# ---------------------------------------------------------------------------
print("\n[Step 5] Selecting Best Candidate and Optimizing Decision Threshold...")
# Candidates with highest ranking performance (ROC-AUC and PR-AUC) and best balanced F1/Recall
# Let's inspect top candidates
top_model_name = cv_df.iloc[0]["Model"]
print(f"Top candidate by CV F1-Score: {top_model_name}")

# Also compare against high-ranking tree models like XGBoost / CatBoost / Logistic Regression
best_oof_probs = oof_predictions[top_model_name]

# Threshold search across OOF predictions
thresholds = np.linspace(0.10, 0.90, 81)
threshold_metrics = []

for t in thresholds:
    preds_t = (best_oof_probs >= t).astype(int)
    cm_t = confusion_matrix(y_train, preds_t)
    rec_t = recall_score(y_train, preds_t)
    prec_t = precision_score(y_train, preds_t, zero_division=0)
    f1_t = f1_score(y_train, preds_t, zero_division=0)
    flagged_t = int(preds_t.sum())
    flagged_pct = (flagged_t / len(y_train)) * 100

    threshold_metrics.append({
        "threshold": round(float(t), 4),
        "recall": round(float(rec_t), 4),
        "precision": round(float(prec_t), 4),
        "f1_score": round(float(f1_t), 4),
        "accuracy": round(float(accuracy_score(y_train, preds_t)), 4),
        "tn": int(cm_t[0, 0]),
        "fp": int(cm_t[0, 1]),
        "fn": int(cm_t[1, 0]),
        "tp": int(cm_t[1, 1]),
        "flagged_customers": flagged_t,
        "flagged_pct": round(flagged_pct, 2),
    })

thresh_df = pd.DataFrame(threshold_metrics)
thresh_df.to_csv(EXP_DIR / "threshold_optimization.csv", index=False)

# Optimal threshold by maximum F1-score on OOF
best_f1_idx = thresh_df["f1_score"].idxmax()
best_f1_row = thresh_df.iloc[best_f1_idx]
optimal_threshold = best_f1_row["threshold"]

print(f"Optimal F1 Threshold (from OOF): {optimal_threshold:.2f}")
print(f"  At threshold {optimal_threshold:.2f}: Recall={best_f1_row['recall']:.4f}, Prec={best_f1_row['precision']:.4f}, F1={best_f1_row['f1_score']:.4f}, Flagged={best_f1_row['flagged_customers']} ({best_f1_row['flagged_pct']}%)")

# Comparison with default 0.50 threshold on OOF
default_t_row = thresh_df[thresh_df["threshold"] == 0.50].iloc[0]
print(f"  At threshold 0.50: Recall={default_t_row['recall']:.4f}, Prec={default_t_row['precision']:.4f}, F1={default_t_row['f1_score']:.4f}, Flagged={default_t_row['flagged_customers']} ({default_t_row['flagged_pct']}%)")

# ---------------------------------------------------------------------------
# 6. PROBABILITY CALIBRATION ANALYSIS (OOF)
# ---------------------------------------------------------------------------
print("\n[Step 6] Evaluating Probability Calibration on Out-of-Fold Predictions...")
brier_loss = brier_score_loss(y_train, best_oof_probs)
prob_true, prob_pred = calibration_curve(y_train, best_oof_probs, n_bins=10)

print(f"Brier Score Loss (OOF): {brier_loss:.4f} (Closer to 0 is better; 0.25 is random guessing for 50/50)")

calibration_report = {
    "model": top_model_name,
    "brier_score_loss": round(float(brier_loss), 4),
    "fraction_of_positives": [round(float(p), 4) for p in prob_true],
    "mean_predicted_value": [round(float(p), 4) for p in prob_pred],
}
with open(EXP_DIR / "calibration_metrics.json", "w") as f:
    json.dump(calibration_report, f, indent=2)

# ---------------------------------------------------------------------------
# 7. FINAL EVALUATION ON UNTOUCHED HOLDOUT TEST SET (ONLY ONCE)
# ---------------------------------------------------------------------------
print("\n[Step 7] Final Holdout Test Set Evaluation (Executed Once)...")
# Train winning pipeline on full training partition
winning_preprocessor, winning_estimator = candidates[top_model_name]
winning_pipeline = Pipeline([
    ("preprocessor", winning_preprocessor),
    ("classifier", winning_estimator),
])

winning_pipeline.fit(X_train_raw, y_train)

# Evaluate on X_test_raw
test_probs = winning_pipeline.predict_proba(X_test_raw)[:, 1]

# Default threshold (0.50)
test_preds_default = (test_probs >= 0.50).astype(int)
cm_default = confusion_matrix(y_test, test_preds_default)

# Optimized threshold
test_preds_opt = (test_probs >= optimal_threshold).astype(int)
cm_opt = confusion_matrix(y_test, test_preds_opt)

final_evaluation_results = {
    "model_name": top_model_name,
    "holdout_sample_size": len(y_test),
    "positive_churn_samples": int(y_test.sum()),
    "default_threshold_0_50": {
        "threshold": 0.50,
        "accuracy": round(float(accuracy_score(y_test, test_preds_default)), 4),
        "precision": round(float(precision_score(y_test, test_preds_default, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, test_preds_default)), 4),
        "f1_score": round(float(f1_score(y_test, test_preds_default)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, test_probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, test_probs)), 4),
        "confusion_matrix": cm_default.tolist(),
        "tn": int(cm_default[0, 0]),
        "fp": int(cm_default[0, 1]),
        "fn": int(cm_default[1, 0]),
        "tp": int(cm_default[1, 1]),
    },
    "optimized_threshold": {
        "threshold": float(optimal_threshold),
        "accuracy": round(float(accuracy_score(y_test, test_preds_opt)), 4),
        "precision": round(float(precision_score(y_test, test_preds_opt, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, test_preds_opt)), 4),
        "f1_score": round(float(f1_score(y_test, test_preds_opt)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, test_probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, test_probs)), 4),
        "confusion_matrix": cm_opt.tolist(),
        "tn": int(cm_opt[0, 0]),
        "fp": int(cm_opt[0, 1]),
        "fn": int(cm_opt[1, 0]),
        "tp": int(cm_opt[1, 1]),
    },
}

with open(EXP_DIR / "final_holdout_evaluation.json", "w") as f:
    json.dump(final_evaluation_results, f, indent=2)

print("\nFinal Holdout Results:")
print("  Default (0.50):")
print(f"    Recall: {final_evaluation_results['default_threshold_0_50']['recall']:.4f} | Prec: {final_evaluation_results['default_threshold_0_50']['precision']:.4f} | F1: {final_evaluation_results['default_threshold_0_50']['f1_score']:.4f} | ROC-AUC: {final_evaluation_results['default_threshold_0_50']['roc_auc']:.4f}")
print("  Optimized Threshold:")
print(f"    Threshold: {optimal_threshold:.2f} | Recall: {final_evaluation_results['optimized_threshold']['recall']:.4f} | Prec: {final_evaluation_results['optimized_threshold']['precision']:.4f} | F1: {final_evaluation_results['optimized_threshold']['f1_score']:.4f} | ROC-AUC: {final_evaluation_results['optimized_threshold']['roc_auc']:.4f}")

# Save winning pipeline to experiments folder safely (DO NOT overwrite original churn_model.pkl)
joblib.dump(winning_pipeline, EXP_DIR / "selected_model_pipeline.pkl")

# Save selected model configuration
model_config = {
    "model_name": top_model_name,
    "optimal_threshold": float(optimal_threshold),
    "features_numeric": numeric_features,
    "features_categorical": categorical_features,
    "pipeline_steps": [name for name, _ in winning_pipeline.steps],
    "classifier_params": winning_pipeline.named_steps["classifier"].get_params(),
}
with open(EXP_DIR / "selected_model_config.json", "w") as f:
    # Handle non-serializable objects
    json.dump(model_config, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# 8. VISUALIZATIONS & PLOTS
# ---------------------------------------------------------------------------
print("\n[Step 8] Generating evaluation plots...")

# Plot 1: Threshold Trade-off Curves
plt.figure(figsize=(9, 5))
plt.plot(thresh_df["threshold"], thresh_df["recall"], label="Recall", color="#dc2626", lw=2)
plt.plot(thresh_df["threshold"], thresh_df["precision"], label="Precision", color="#2563eb", lw=2)
plt.plot(thresh_df["threshold"], thresh_df["f1_score"], label="F1-Score", color="#16a34a", lw=2)
plt.axvline(optimal_threshold, color="#16a34a", linestyle="--", label=f"Optimal F1 (t={optimal_threshold:.2f})")
plt.axvline(0.50, color="#6b7280", linestyle=":", label="Default (t=0.50)")
plt.title(f"Threshold Optimization Curve ({top_model_name})", fontsize=12, fontweight="bold")
plt.xlabel("Decision Threshold", fontsize=10)
plt.ylabel("Score", fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(EXP_DIR / "threshold_tradeoffs.png", dpi=150)
plt.close()

# Plot 2: ROC & Precision-Recall Curves
fpr, tpr, _ = roc_curve(y_test, test_probs)
prec_curve, rec_curve, _ = precision_recall_curve(y_test, test_probs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
# ROC
ax1.plot(fpr, tpr, color="#4f46e5", lw=2, label=f"ROC (AUC = {roc_auc_score(y_test, test_probs):.4f})")
ax1.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--")
ax1.set_title("Receiver Operating Characteristic (ROC)", fontweight="bold")
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# PR
ax2.plot(rec_curve, prec_curve, color="#059669", lw=2, label=f"PR (AUC = {average_precision_score(y_test, test_probs):.4f})")
ax2.axhline(y_test.mean(), color="#9ca3af", linestyle="--", label=f"Baseline ({y_test.mean():.2f})")
ax2.set_title("Precision-Recall Curve", fontweight="bold")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend(loc="lower left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(EXP_DIR / "roc_pr_curves.png", dpi=150)
plt.close()

# Plot 3: Calibration Curve
plt.figure(figsize=(7, 6))
plt.plot(prob_pred, prob_true, marker="o", color="#4f46e5", label=f"OOF Reliability (Brier={brier_loss:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="#9ca3af", label="Perfect Calibration")
plt.title(f"Probability Calibration Curve ({top_model_name})", fontsize=12, fontweight="bold")
plt.xlabel("Mean Predicted Probability", fontsize=10)
plt.ylabel("Fraction of True Churners", fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(EXP_DIR / "calibration_curve.png", dpi=150)
plt.close()

# Plot 4: Confusion Matrix Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
ax1.matshow(cm_default, cmap="Blues", alpha=0.6)
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(cm_default[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
ax1.set_title("Default Threshold (0.50)", fontweight="bold", pad=15)
ax1.set_xticks([0, 1], ["Stay (0)", "Churn (1)"])
ax1.set_yticks([0, 1], ["Stay (0)", "Churn (1)"])
ax1.set_ylabel("Actual")
ax1.set_xlabel("Predicted")

ax2.matshow(cm_opt, cmap="Greens", alpha=0.6)
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm_opt[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
ax2.set_title(f"Optimized Threshold ({optimal_threshold:.2f})", fontweight="bold", pad=15)
ax2.set_xticks([0, 1], ["Stay (0)", "Churn (1)"])
ax2.set_yticks([0, 1], ["Stay (0)", "Churn (1)"])
ax2.set_ylabel("Actual")
ax2.set_xlabel("Predicted")

plt.tight_layout()
plt.savefig(EXP_DIR / "confusion_matrices.png", dpi=150)
plt.close()

print(f"\nAll Phase 2 experimental benchmarks complete! Output saved safely in: {EXP_DIR}")
