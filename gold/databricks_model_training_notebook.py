# Databricks notebook source
# MAGIC %md
# MAGIC # Model Training 
# MAGIC
# MAGIC **Models trained:**
# MAGIC - Logistic Regression (baseline)
# MAGIC - XGBoost (primary)

# COMMAND ----------

pip install xgboost

# COMMAND ----------

import pandas as pd
import numpy as np
import pickle, os

from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier

RANDOM_SEED = 42
TRAIN_RATIO = 0.75

FEATURE_COLS = [
    # Cost group
    "billing_ratio",      # billed_amount / expected_cost  (computed from cost join)
    "cost_diff",          # billed_amount - expected_cost
    "high_cost_flag",     # 1 if billing_ratio > 1.5
    # Provider group 
    "provider_claim_count",    # total claims this provider submitted
    "provider_specialty_enc",  # label-encoded specialty
    # Diagnosis group
    "severity_score",     # Low=1 / Medium=2 / High=3
    "diag_claim_count",   # frequency of this diagnosis code
    "diag_category_enc",  # label-encoded diagnosis category
    # Claim integrity group
    "is_billed_missing",  # 1 if billed_amount was null in bronze
    "is_proc_missing",    # 1 if procedure_code was null in bronze
    "is_diag_missing",    # 1 if diagnosis_code was null in bronze
    # Temporal
    "claim_age_days",     # days since claim date
]
# EXCLUDED (target leakage — see feature_engineering notebook comments):
#   pre_risk_score, final_risk_score, risk_tier_enc,
#   provider_denial_rate, provider_risk_score
TARGET_COL = "denial_flag"
ID_COL     = "claim_id"

print(f"Train/Test split : {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}")
print(f"Features         : {len(FEATURE_COLS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load gold_claim_features

# COMMAND ----------

feat_df = spark.table("workspace.gold.gold_claim_features")
print(f"gold_claim_features: {feat_df.count():,} rows | {len(feat_df.columns)} columns")

# Convert to Pandas (dataset ~1K rows — fits in driver memory)
feat_pd = feat_df.toPandas()

X = feat_pd[FEATURE_COLS] 
y = feat_pd[TARGET_COL]

print(f"\nClass distribution:")
print(f"  Denied   (0): {int((y==0).sum()):,}  ({100*(y==0).mean():.1f}%)")
print(f"  Approved (1): {int((y==1).sum()):,}  ({100*(y==1).mean():.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train / Test Split

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=(1 - TRAIN_RATIO), 
    random_state=RANDOM_SEED,
    stratify=y       # preserve class balance in both splits
)

print(f"Train : {len(X_train):,} rows  | denied={int((y_train==0).sum())}  approved={int((y_train==1).sum())}")
print(f"Test  : {len(X_test):,} rows   | denied={int((y_test==0).sum())}  approved={int((y_test==1).sum())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 1: Logistic Regression (Baseline)
# MAGIC
# MAGIC - Scaled features (StandardScaler)
# MAGIC - class_weight="balanced" handles label imbalance
# MAGIC - Gives per-feature coefficients for explainability

# COMMAND ----------

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_SEED,
    class_weight="balanced",
    solver="lbfgs"
)
lr.fit(X_train_sc, y_train)

y_pred_lr = lr.predict(X_test_sc)
y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]

print("=" * 50)
print("  LOGISTIC REGRESSION")
print("=" * 50)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred_lr)*100:.2f}%")
print(f"  Precision : {precision_score(y_test, y_pred_lr, zero_division=0)*100:.2f}%")
print(f"  Recall    : {recall_score(y_test, y_pred_lr, zero_division=0)*100:.2f}%")
print(f"  F1 Score  : {f1_score(y_test, y_pred_lr, zero_division=0)*100:.2f}%")
print()
print(classification_report(y_test, y_pred_lr,
                            target_names=["Denied(0)", "Approved(1)"],
                            zero_division=0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 2: XGBoost (Primary)
# MAGIC
# MAGIC - Gradient boosted trees — handles non-linear interactions natively
# MAGIC - No scaling needed
# MAGIC - tree_method="hist" — fast, memory-efficient on CPU

# COMMAND ----------

xgb = XGBClassifier(
    n_estimators     = 200,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = "logloss",
    random_state     = RANDOM_SEED,
    tree_method      = "hist",
)
xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

print("=" * 50)
print("  XGBOOST CLASSIFIER")
print("=" * 50)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred_xgb)*100:.2f}%")
print(f"  Precision : {precision_score(y_test, y_pred_xgb, zero_division=0)*100:.2f}%")
print(f"  Recall    : {recall_score(y_test, y_pred_xgb, zero_division=0)*100:.2f}%")
print(f"  F1 Score  : {f1_score(y_test, y_pred_xgb, zero_division=0)*100:.2f}%")
print()
print(classification_report(y_test, y_pred_xgb,
                            target_names=["Denied(0)", "Approved(1)"],
                            zero_division=0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model 2b: XGBoost Fine-Tuning (GridSearchCV)

# COMMAND ----------

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
import warnings

# COMMAND ----------

warnings.filterwarnings("ignore")

# ── 1. Define the parameter grid ────────────────────────────────────────────
param_grid = {
    "n_estimators"     : [100, 200, 300],
    "max_depth"        : [3, 4, 5, 6],
    "learning_rate"    : [0.01, 0.05, 0.1],
    "subsample"        : [0.7, 0.8, 1.0],
    "colsample_bytree" : [0.7, 0.8, 1.0],
    "min_child_weight" : [1, 3, 5],      # controls overfitting
    "gamma"            : [0, 0.1, 0.3],  # min loss reduction to split
    "reg_alpha"        : [0, 0.1, 0.5],  # L1 regularisation
    "reg_lambda"       : [1, 1.5, 2.0],  # L2 regularisation
}

# ── 2. Base estimator (same fixed params as your original) ──────────────────
base_xgb = XGBClassifier(
    eval_metric  = "logloss",
    random_state = RANDOM_SEED,
    tree_method  = "hist",
)

# ── 3. Stratified K-Fold (preserves class balance across folds) ─────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# ── 4. Scorer — use F1 (better than accuracy for imbalanced labels) ─────────
scorer = make_scorer(f1_score, zero_division=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option A — GridSearchCV (exhaustive, slower)

# COMMAND ----------

grid_search = GridSearchCV(
    estimator  = base_xgb,
    param_grid = param_grid,
    scoring    = scorer,
    cv         = cv,
    n_jobs     = -1,        # use all CPU cores
    verbose    = 2,
    refit      = True       # auto-refit best model on full train set
)

grid_search.fit(X_train, y_train)

print(f"\nBest CV F1     : {grid_search.best_score_*100:.2f}%")
print(f"Best Params    : {grid_search.best_params_}")

# COMMAND ----------

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator          = base_xgb,
    param_distributions= param_grid,
    n_iter             = 50,           # number of random combos to try
    scoring            = scorer,
    cv                 = cv,
    n_jobs             = -1,
    verbose            = 2,
    random_state       = RANDOM_SEED,
    refit              = True
)

random_search.fit(X_train, y_train)

print(f"\nBest CV F1     : {random_search.best_score_*100:.2f}%")
print(f"Best Params    : {random_search.best_params_}")

# COMMAND ----------

best_xgb = random_search.best_estimator_

y_pred_tuned = best_xgb.predict(X_test)
y_prob_tuned = best_xgb.predict_proba(X_test)[:, 1]

print("=" * 55)
print("  XGBOOST — TUNED")
print("=" * 55)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred_tuned)*100:.2f}%")
print(f"  Precision : {precision_score(y_test, y_pred_tuned, zero_division=0)*100:.2f}%")
print(f"  Recall    : {recall_score(y_test, y_pred_tuned, zero_division=0)*100:.2f}%")
print(f"  F1 Score  : {f1_score(y_test, y_pred_tuned, zero_division=0)*100:.2f}%")
print(f"  ROC-AUC   : {roc_auc_score(y_test, y_prob_tuned)*100:.2f}%")
print()
print(classification_report(y_test, y_pred_tuned,
                            target_names=["Denied(0)", "Approved(1)"],
                            zero_division=0))
