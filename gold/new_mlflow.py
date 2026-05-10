# Databricks notebook source
# MAGIC %md
# MAGIC # Step 10B — Model Training & Hyperparameter Tuning
# MAGIC
# MAGIC ## Databricks Community Edition Compatible Version
# MAGIC
# MAGIC This notebook:
# MAGIC - trains Logistic Regression + XGBoost
# MAGIC - performs hyperparameter tuning
# MAGIC - logs metrics/models using MLflow
# MAGIC - avoids Model Registry errors in Databricks CE
# MAGIC - serves models directly using `runs:/` URIs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Imports & MLflow Configuration

# COMMAND ----------

pip install xgboost

# COMMAND ----------

import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    make_scorer
)

from xgboost import XGBClassifier

# ---------------- CONFIG ----------------

RANDOM_SEED = 42
TRAIN_RATIO = 0.80
N_FOLDS     = 5

# ---------------- MLflow Setup ----------------
# Databricks CE SAFE CONFIG

MLFLOW_DIR   = "/tmp/mlruns"
TRACKING_URI = f"file://{MLFLOW_DIR}"
REGISTRY_URI = f"file://{MLFLOW_DIR}"

EXPERIMENT_NAME = "claim_denial_prevention_mlflow"

# Environment variables
os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URI
os.environ["MLFLOW_REGISTRY_URI"] = REGISTRY_URI

# Explicitly configure MLflow
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(REGISTRY_URI)

# Create experiment if not exists
try:
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if exp is None:
        mlflow.create_experiment(EXPERIMENT_NAME)

    mlflow.set_experiment(EXPERIMENT_NAME)

except Exception as e:
    print(f"MLflow setup warning: {e}")

print("=" * 60)
print("MLFLOW CONFIGURATION")
print("=" * 60)
print(f"Tracking URI : {TRACKING_URI}")
print(f"Registry URI : {REGISTRY_URI}")
print(f"Experiment   : {EXPERIMENT_NAME}")
print(f"Train/Test   : {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}")
print(f"CV Folds     : {N_FOLDS}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Load Gold Feature Table

# COMMAND ----------

feat_spark = spark.table("workspace.gold.gold_claim_features")
feat_pd = feat_spark.toPandas()

FEATURE_COLS = [
    "billing_ratio",
    "cost_diff",
    "high_cost_flag",
    "provider_claim_count",
    "provider_specialty_enc",
    "severity_score",
    "diag_claim_count",
    "is_billed_missing",
    "is_proc_missing",
    "is_diag_missing",
    "claim_age_days",
]

TARGET_COL = "denial_flag"
ID_COL     = "claim_id"

X = feat_pd[FEATURE_COLS]
y = feat_pd[TARGET_COL]

total = len(y)

denied   = int((y == 0).sum())
approved = int((y == 1).sum())

print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"Dataset Shape : {X.shape}")
print(f"Denied Claims : {denied:,}")
print(f"Approved      : {approved:,}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Train/Test Split + Scaling

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=(1 - TRAIN_RATIO),
    stratify=y,
    random_state=RANDOM_SEED
)

print("=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print(
    f"Train : {len(X_train):,} rows | "
    f"Denied={(y_train==0).sum()} "
    f"Approved={(y_train==1).sum()}"
)

print(
    f"Test  : {len(X_test):,} rows | "
    f"Denied={(y_test==0).sum()} "
    f"Approved={(y_test==1).sum()}"
)

print("=" * 60)

# Scaling for Logistic Regression
scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Logistic Regression Baseline

# COMMAND ----------

with mlflow.start_run(run_name="logistic_regression") as lr_run:

    lr_params = {
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "C": 1.0
    }

    mlflow.log_params(lr_params)

    lr = LogisticRegression(**lr_params)

    lr.fit(X_train_sc, y_train)

    y_pred_lr = lr.predict(X_test_sc)
    y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]

    acc_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr  = f1_score(
        y_test,
        y_pred_lr,
        zero_division=0,
        pos_label=0
    )

    auc_lr = roc_auc_score(y_test, y_prob_lr)

    mlflow.log_metric("test_accuracy", acc_lr)
    mlflow.log_metric("test_f1", f1_lr)
    mlflow.log_metric("test_roc_auc", auc_lr)

    mlflow.sklearn.log_model(
        lr,
        artifact_path="model",
        input_example=X_test.iloc[:3]
    )

    lr_run_id = lr_run.info.run_id

    print("=" * 60)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 60)
    print(f"Accuracy : {acc_lr*100:.2f}%")
    print(f"F1 Score : {f1_lr:.4f}")
    print(f"ROC-AUC  : {auc_lr:.4f}")
    print(f"Run ID   : {lr_run_id}")
    print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — XGBoost Hyperparameter Tuning

# COMMAND ----------

with mlflow.start_run(run_name="xgboost_tuned") as xgb_run:

    scale_pos = approved / denied

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [1, 1.5, 2.0],
    }

    base_xgb = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        tree_method="hist",
        scale_pos_weight=scale_pos
    )

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    scorer = make_scorer(
        f1_score,
        zero_division=0,
        pos_label=0
    )

    random_search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_grid,
        n_iter=20,
        scoring=scorer,
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        refit=True
    )

    print("=" * 60)
    print("RUNNING RANDOMIZED SEARCH CV")
    print("=" * 60)

    random_search.fit(X_train, y_train)

    print("\nBest Parameters:")
    print(random_search.best_params_)

    print(f"\nBest CV F1: {random_search.best_score_:.4f}")

    best_xgb = random_search.best_estimator_

    mlflow.log_params(random_search.best_params_)

    y_pred_xgb = best_xgb.predict(X_test)
    y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]

    acc_xgb = accuracy_score(y_test, y_pred_xgb)

    prec_xgb = precision_score(
        y_test,
        y_pred_xgb,
        zero_division=0,
        pos_label=0
    )

    rec_xgb = recall_score(
        y_test,
        y_pred_xgb,
        zero_division=0,
        pos_label=0
    )

    f1_xgb = f1_score(
        y_test,
        y_pred_xgb,
        zero_division=0,
        pos_label=0
    )

    auc_xgb = roc_auc_score(y_test, y_prob_xgb)

    mlflow.log_metric("test_accuracy", acc_xgb)
    mlflow.log_metric("test_precision", prec_xgb)
    mlflow.log_metric("test_recall", rec_xgb)
    mlflow.log_metric("test_f1", f1_xgb)
    mlflow.log_metric("test_roc_auc", auc_xgb)

    mlflow.xgboost.log_model(
        best_xgb,
        artifact_path="model",
        input_example=X_test.iloc[:3]
    )

    xgb_run_id = xgb_run.info.run_id

    print("=" * 60)
    print("TUNED XGBOOST RESULTS")
    print("=" * 60)

    print(f"Accuracy  : {acc_xgb*100:.2f}%")
    print(f"Precision : {prec_xgb:.4f}")
    print(f"Recall    : {rec_xgb:.4f}")
    print(f"F1 Score  : {f1_xgb:.4f}")
    print(f"ROC-AUC   : {auc_xgb:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            y_pred_xgb,
            target_names=["Denied(0)", "Approved(1)"],
            zero_division=0
        )
    )

    print(f"Run ID: {xgb_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Select Best Model

# COMMAND ----------

if acc_xgb >= acc_lr:

    best_run_id = xgb_run_id
    best_model_name = "XGBoost (Tuned)"

else:

    best_run_id = lr_run_id
    best_model_name = "Logistic Regression"

prod_model_uri = f"runs:/{best_run_id}/model"

print("=" * 60)
print("BEST MODEL SELECTION")
print("=" * 60)
print(f"Selected Model : {best_model_name}")
print(f"Model URI      : {prod_model_uri}")
print("=" * 60)

# Load model for inference
if best_model_name == "XGBoost (Tuned)":

    prod_model = mlflow.xgboost.load_model(prod_model_uri)

    def predict_fn(data):
        preds = prod_model.predict(data)
        probs = prod_model.predict_proba(data)[:, 1]
        return preds, probs

else:

    prod_model = mlflow.sklearn.load_model(prod_model_uri)

    def predict_fn(data):
        data_sc = scaler.transform(data)
        preds = prod_model.predict(data_sc)
        probs = prod_model.predict_proba(data_sc)[:, 1]
        return preds, probs

print("✅ Production model loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Inference on New Claims

# COMMAND ----------

new_claims_data = pd.DataFrame([

    {
        "billing_ratio": 1.1,
        "cost_diff": 500,
        "high_cost_flag": 0,
        "provider_claim_count": 45,
        "provider_specialty_enc": 2,
        "severity_score": 2,
        "diag_claim_count": 150,
        "diagnosis_category_enc": 1,
        "is_billed_missing": 0,
        "is_proc_missing": 0,
        "is_diag_missing": 0,
        "claim_age_days": 10,
    },

    {
        "billing_ratio": 3.5,
        "cost_diff": 15000,
        "high_cost_flag": 1,
        "provider_claim_count": 5,
        "provider_specialty_enc": 1,
        "severity_score": 1,
        "diag_claim_count": 20,
        "diagnosis_category_enc": 3,
        "is_billed_missing": 0,
        "is_proc_missing": 1,
        "is_diag_missing": 0,
        "claim_age_days": 45,
    },

    {
        "billing_ratio": 0.9,
        "cost_diff": -100,
        "high_cost_flag": 0,
        "provider_claim_count": 200,
        "provider_specialty_enc": 0,
        "severity_score": 3,
        "diag_claim_count": 500,
        "diagnosis_category_enc": 2,
        "is_billed_missing": 0,
        "is_proc_missing": 0,
        "is_diag_missing": 0,
        "claim_age_days": 2,
    }
])

print("=" * 60)
print("INFERENCE ON NEW CLAIMS")
print("=" * 60)

new_preds, new_probs = predict_fn(new_claims_data)

for i, (pred, prob) in enumerate(zip(new_preds, new_probs)):

    deny_probability = 1 - prob

    decision = "DENIED" if pred == 0 else "APPROVED"

    print(f"\nSample Claim {i+1}")
    print("-" * 40)
    print(f"Denial Probability : {deny_probability*100:.2f}%")
    print(f"Final Decision     : {decision}")

print("\n✅ Inference completed successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 — MLflow UI Information

# COMMAND ----------

print("=" * 60)
print("MLFLOW ARTIFACT INFORMATION")
print("=" * 60)

print(f"Best Model URI:")
print(prod_model_uri)

print("\nUse this URI for serving/loading models later.")

print("\nExample:")
print("mlflow.pyfunc.load_model(model_uri)")

print("=" * 60)
