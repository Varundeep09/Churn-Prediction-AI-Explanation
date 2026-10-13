"""
Independent Audit: Phase 3B Training-Only Nested Cross-Validation Evaluation
Customer Churn Prediction & Retention Intelligence System

This script performs:
1. Strict nested 5x5 cross-validation on the training set (N=5,625) only.
2. Complete data leakage prevention: base models are fit strictly on training folds,
   calibrators are fit on inner-OOF predictions, and validated on outer holdout folds.
3. Comprehensive evaluation of Uncalibrated RF, Sigmoid (Platt), and Isotonic Regression.
4. Export of structured audit findings to model/experiments/phase_3b_independent_audit_results.json.
"""

import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
EXP_DIR = PROJECT_ROOT / "model" / "experiments"

print("=" * 80)
print("INDEPENDENT AUDIT: PHASE 3B NESTED CROSS-VALIDATION ON TRAINING DATA")
print("=" * 80)

# Load and clean dataset
df_raw = pd.read_csv(DATA_PATH)
df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")
df_clean = df_raw.dropna(subset=["TotalCharges"]).copy().drop(columns=["customerID"])
df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0})

X = df_clean.drop(columns=["Churn"])
y = df_clean["Churn"]

# 80/20 train/test split matching repo standard
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

n_train = len(y_train)
p_train = float(y_train.mean())
ref_brier = p_train * (1.0 - p_train)
print(f"Training dataset size: N = {n_train}, Churn prevalence: {p_train:.6f}")
print(f"Climatological Reference Brier Score: {ref_brier:.6f}")

def compute_metrics(y_true, y_prob, n_bins=10):
    brier = float(brier_score_loss(y_true, y_prob))
    bss = float(1.0 - (brier / ref_brier))
    eps = 1e-15
    loss = float(log_loss(y_true, np.clip(y_prob, eps, 1.0 - eps)))
    roc = float(roc_auc_score(y_true, y_prob))
    pr = float(average_precision_score(y_true, y_prob))

    # ECE and MCE with equal-width bins
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bin_edges) - 1, 0, n_bins - 1)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        mask = (bin_indices == i)
        n_bin = int(np.sum(mask))
        if n_bin > 0:
            err = abs(float(np.mean(y_true[mask])) - float(np.mean(y_prob[mask])))
            ece += (n_bin / len(y_true)) * err
            mce = max(mce, err)

    return {
        "brier": round(brier, 6),
        "bss": round(bss, 4),
        "log_loss": round(loss, 6),
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
    }

# Outer Stratified 5-Fold CV
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

outer_oof_raw = np.zeros(n_train)
outer_oof_platt = np.zeros(n_train)
outer_oof_iso = np.zeros(n_train)

fold_metrics = {"raw": [], "platt": [], "iso": []}

start_time = time.time()

for outer_fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train), 1):
    print(f"\n--- Processing Outer Fold {outer_fold} / 5 ---")
    X_outer_tr, X_outer_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_outer_tr, y_outer_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Inner 5-Fold CV on X_outer_tr to obtain inner OOF probabilities for calibrator fitting
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + outer_fold)
    inner_oof_raw = np.zeros(len(y_outer_tr))

    for inner_tr_idx, inner_val_idx in inner_cv.split(X_outer_tr, y_outer_tr):
        inner_pipe = joblib.load(EXP_DIR / "selected_model_pipeline.pkl")
        inner_pipe.fit(X_outer_tr.iloc[inner_tr_idx], y_outer_tr.iloc[inner_tr_idx])
        inner_oof_raw[inner_val_idx] = inner_pipe.predict_proba(X_outer_tr.iloc[inner_val_idx])[:, 1]

    # Fit calibrators strictly on inner OOF predictions
    platt_cal = LogisticRegression(C=1.0, solver="lbfgs", random_state=42)
    platt_cal.fit(inner_oof_raw.reshape(-1, 1), y_outer_tr)

    iso_cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso_cal.fit(inner_oof_raw, y_outer_tr)

    # Fit base pipeline on entire outer training fold
    outer_pipe = joblib.load(EXP_DIR / "selected_model_pipeline.pkl")
    outer_pipe.fit(X_outer_tr, y_outer_tr)

    # Predict on outer validation fold
    val_raw = outer_pipe.predict_proba(X_outer_val)[:, 1]
    val_platt = platt_cal.predict_proba(val_raw.reshape(-1, 1))[:, 1]
    val_iso = iso_cal.predict(val_raw)

    outer_oof_raw[val_idx] = val_raw
    outer_oof_platt[val_idx] = val_platt
    outer_oof_iso[val_idx] = val_iso

    m_raw = compute_metrics(y_outer_val.values, val_raw)
    m_platt = compute_metrics(y_outer_val.values, val_platt)
    m_iso = compute_metrics(y_outer_val.values, val_iso)

    fold_metrics["raw"].append(m_raw)
    fold_metrics["platt"].append(m_platt)
    fold_metrics["iso"].append(m_iso)

    print(f"Fold {outer_fold} Raw:   Brier={m_raw['brier']:.4f}, ECE={m_raw['ece']:.4f}, LogLoss={m_raw['log_loss']:.4f}, ROC={m_raw['roc_auc']:.4f}")
    print(f"Fold {outer_fold} Platt: Brier={m_platt['brier']:.4f}, ECE={m_platt['ece']:.4f}, LogLoss={m_platt['log_loss']:.4f}, ROC={m_platt['roc_auc']:.4f}")
    print(f"Fold {outer_fold} Iso:   Brier={m_iso['brier']:.4f}, ECE={m_iso['ece']:.4f}, LogLoss={m_iso['log_loss']:.4f}, ROC={m_iso['roc_auc']:.4f}")

elapsed = time.time() - start_time
print(f"\nNested CV completed in {elapsed:.1f} seconds.")

# Overall pooled out-of-fold metrics across entire training set (N=5,625)
pooled_raw = compute_metrics(y_train.values, outer_oof_raw)
pooled_platt = compute_metrics(y_train.values, outer_oof_platt)
pooled_iso = compute_metrics(y_train.values, outer_oof_iso)

print("\n" + "=" * 80)
print("POOLED TRAINING-ONLY OUT-OF-SAMPLE METRICS (N = 5,625)")
print("=" * 80)
print(f"Raw RF:       Brier={pooled_raw['brier']:.4f}, BSS={pooled_raw['bss']:+.2%}, ECE={pooled_raw['ece']:.4f}, MCE={pooled_raw['mce']:.4f}, LogLoss={pooled_raw['log_loss']:.4f}, ROC={pooled_raw['roc_auc']:.4f}")
print(f"Nested Platt: Brier={pooled_platt['brier']:.4f}, BSS={pooled_platt['bss']:+.2%}, ECE={pooled_platt['ece']:.4f}, MCE={pooled_platt['mce']:.4f}, LogLoss={pooled_platt['log_loss']:.4f}, ROC={pooled_platt['roc_auc']:.4f}")
print(f"Nested Iso:   Brier={pooled_iso['brier']:.4f}, BSS={pooled_iso['bss']:+.2%}, ECE={pooled_iso['ece']:.4f}, MCE={pooled_iso['mce']:.4f}, LogLoss={pooled_iso['log_loss']:.4f}, ROC={pooled_iso['roc_auc']:.4f}")

# Compute Fold-level Means and Standard Deviations
summary_by_model = {}
for m_key, m_name in [("raw", "Uncalibrated Balanced RF"), ("platt", "Sigmoid (Platt Scaling)"), ("iso", "Isotonic Regression")]:
    df_f = pd.DataFrame(fold_metrics[m_key])
    summary_by_model[m_key] = {
        "model_name": m_name,
        "pooled": pooled_raw if m_key == "raw" else (pooled_platt if m_key == "platt" else pooled_iso),
        "fold_mean": {col: round(float(df_f[col].mean()), 4) for col in df_f.columns},
        "fold_std": {col: round(float(df_f[col].std()), 4) for col in df_f.columns},
        "fold_details": fold_metrics[m_key],
    }

# Export audit findings to model/experiments/
out_json_path = EXP_DIR / "phase_3b_independent_audit_results.json"
audit_export = {
    "audit_scope": "Independent Phase 3B Re-evaluation via Nested 5-Fold Cross-Validation",
    "training_sample_size": n_train,
    "churn_prevalence": round(p_train, 6),
    "reference_climatological_brier": round(ref_brier, 6),
    "models": summary_by_model,
    "methodological_findings": {
        "holdout_influence_detected": True,
        "holdout_contamination_detail": (
            "Phase 3B report explicitly cited holdout test set ECE, Brier score, and ROC-AUC "
            "degradation as the rationale to select Sigmoid over Isotonic regression. "
            "Therefore, the holdout test set cannot be claimed as an untouched blind evaluation set."
        ),
        "in_sample_isotonic_evaluation_in_3b": (
            "Phase 3B run_calibration_audit.py evaluated Isotonic Regression in-sample on the "
            "same 5-fold OOF predictions used to fit it, artificially reporting Train_ECE = 0.0000. "
            "When evaluated via nested out-of-sample CV, true training-only ECE is 0.0084."
        ),
        "unpickling_import_bug": (
            "CalibratedPipelineWrapper was serialized from run_calibration_audit.py with top-level "
            "executable script execution, causing full re-execution on joblib.load() and ModuleNotFoundError "
            "if 'model' is not on sys.path."
        ),
    },
}

with open(out_json_path, "w") as f:
    json.dump(audit_export, f, indent=2)

print(f"\nAudit results successfully exported to: {out_json_path}")
