# Databricks notebook source
# MAGIC %md
# MAGIC # Step 10B — Model Training & Hyperparameter Tuning
# MAGIC
# MAGIC **Lineage:**
# MAGIC ```
# MAGIC workspace.default.gold_claim_features ──► [This Notebook] ──► MLflow (Databricks CE)
# MAGIC ```
# MAGIC
# MAGIC ## Design Decisions (Q&A)
# MAGIC
# MAGIC ### 1. Train / Test Split
# MAGIC Since our dataset is 1,000 rows, an 80/20 split gives 800 training rows and 200 test rows. 
# MAGIC We use a stratified split to ensure class balance across both sets.
# MAGIC
# MAGIC ### 2. Hyperparameter Tuning
# MAGIC We use `RandomizedSearchCV` to fine-tune XGBoost. This prevents overfitting by finding the optimal combination of `max_depth`, `min_child_weight`, and regularisation (`reg_lambda`, `reg_alpha`) while improving generalisation metrics (F1 and ROC-AUC).
# MAGIC
# MAGIC ### 3. MLflow in Databricks Community Edition (CE)
# MAGIC Note: Databricks CE does not support the formal "Model Registry" (`mlflow.register_model()`).
# MAGIC Therefore, we log the models to the MLflow tracking server and serve them directly from the `runs:/<run_id>/model` URI.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Imports & Config

# COMMAND ----------

pip install xgboost

# COMMAND ----------

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import warnings 
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, make_scorer
)
from xgboost import XGBClassifier

RANDOM_SEED  = 42
TRAIN_RATIO  = 0.80   # 80/20 split for 1,000 rows
N_FOLDS      = 5      # stratified K-fold for CV
EXPERIMENT   = "/Users/varadnaik03@gmail.com/claim_denial_prevention_mlflow"  # Adjust as needed

# MLflow experiment — creates it if it doesn't exist
try:
    mlflow.set_experiment(EXPERIMENT)
except Exception as e:
    print(f"Warning setting experiment: {e}")

print(f"Train/Test split : {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}")
print(f"CV folds         : {N_FOLDS}")
print(f"MLflow experiment: {EXPERIMENT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Load Feature Table

# COMMAND ----------

feat_spark = spark.table("workspace.gold.gold_claim_features")
feat_pd    = feat_spark.toPandas()

FEATURE_COLS = [
    "billing_ratio",            
    "cost_diff",                
    "high_cost_flag",           
    "provider_claim_count",     
    "provider_specialty_enc",   
    "severity_score",           
    "diag_claim_count",         
    # "diagnosis_category_enc",   # Commented out due to KeyError
    "is_billed_missing",        
    "is_proc_missing",          
    "is_diag_missing",          
    "claim_age_days",           
]
TARGET_COL = "denial_flag"
ID_COL     = "claim_id"

X = feat_pd[FEATURE_COLS]
y = feat_pd[TARGET_COL]

total    = len(y)
denied   = int((y == 0).sum())
approved = int((y == 1).sum())

print(f"Dataset shape    : {X.shape}")
print(f"\nClass distribution:")
print(f"  Denied   (0)  : {denied:,}  ({100*denied/total:.1f}%)")
print(f"  Approved (1)  : {approved:,}  ({100*approved/total:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Train / Test Split (80/20 Stratified)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size   = (1 - TRAIN_RATIO),
    random_state= RANDOM_SEED,
    stratify    = y,
)

print(f"Train : {len(X_train):,} rows  | Denied={int((y_train==0).sum())}  Approved={int((y_train==1).sum())}")
print(f"Test  : {len(X_test):,} rows   | Denied={int((y_test==0).sum())}  Approved={int((y_test==1).sum())}")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Baseline: Logistic Regression

# COMMAND ----------

with mlflow.start_run(run_name="logistic_regression") as lr_run:
    lr_params = dict(
        max_iter     = 1000,
        random_state = RANDOM_SEED,
        class_weight = "balanced", 
        solver       = "lbfgs",
        C            = 1.0,         
    )
    mlflow.log_params(lr_params)

    lr = LogisticRegression(**lr_params)
    lr.fit(X_train_sc, y_train)

    y_pred_lr = lr.predict(X_test_sc)
    y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]

    acc_lr  = accuracy_score(y_test,  y_pred_lr)
    f1_lr   = f1_score(y_test,        y_pred_lr, zero_division=0, pos_label=0)
    auc_lr  = roc_auc_score(y_test,   y_prob_lr)

    mlflow.log_metric("test_accuracy",  acc_lr)
    mlflow.log_metric("test_f1",        f1_lr)
    mlflow.log_metric("test_roc_auc",   auc_lr)

    mlflow.sklearn.log_model(lr, artifact_path="model", input_example=X_test.iloc[:3])
    lr_run_id = lr_run.info.run_id
    print(f"  Logistic Regression ROC-AUC: {auc_lr:.4f}") 
# The error is due to missing input_example for log_model in Databricks CE. Add input_example argument.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — XGBoost: Hyperparameter Tuning
# MAGIC
# MAGIC Using `RandomizedSearchCV` to optimize the model and prevent overfitting.

# COMMAND ----------

with mlflow.start_run(run_name="xgboost_tuned") as xgb_run:
    
    scale_pos = approved / denied
    
    # 1. Define parameter grid for tuning
    param_grid = {
        "n_estimators"     : [100, 200, 300],
        "max_depth"        : [3, 4, 5],
        "learning_rate"    : [0.01, 0.05, 0.1],
        "subsample"        : [0.7, 0.8, 1.0],
        "colsample_bytree" : [0.7, 0.8, 1.0],
        "min_child_weight" : [1, 3, 5],      # controls overfitting
        "gamma"            : [0, 0.1, 0.3],  # min loss reduction to split
        "reg_alpha"        : [0, 0.1, 0.5],  # L1 regularisation
        "reg_lambda"       : [1, 1.5, 2.0],  # L2 regularisation
    }

    base_xgb = XGBClassifier(
        eval_metric      = "logloss",
        random_state     = RANDOM_SEED,
        tree_method      = "hist",
        scale_pos_weight = scale_pos
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scorer = make_scorer(f1_score, zero_division=0, pos_label=0) # optimize for Denied class F1

    random_search = RandomizedSearchCV(
        estimator          = base_xgb,
        param_distributions= param_grid,
        n_iter             = 20,           # limit to 20 for speed
        scoring            = scorer,
        cv                 = cv,
        n_jobs             = -1,
        verbose            = 1,
        random_state       = RANDOM_SEED,
        refit              = True
    )

    print("Running RandomizedSearchCV for XGBoost...")
    random_search.fit(X_train, y_train)
    
    print(f"\nBest CV F1 (Denied) : {random_search.best_score_*100:.2f}%")
    print(f"Best Params         : {random_search.best_params_}")

    # Best model from search
    best_xgb = random_search.best_estimator_
    mlflow.log_params(random_search.best_params_)

    y_pred_xgb = best_xgb.predict(X_test)
    y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]

    acc_xgb  = accuracy_score(y_test,  y_pred_xgb)
    prec_xgb = precision_score(y_test, y_pred_xgb, zero_division=0, pos_label=0)
    rec_xgb  = recall_score(y_test,    y_pred_xgb, zero_division=0, pos_label=0)
    f1_xgb   = f1_score(y_test,        y_pred_xgb, zero_division=0, pos_label=0)
    auc_xgb  = roc_auc_score(y_test,   y_prob_xgb)

    mlflow.log_metric("test_accuracy",  acc_xgb)
    mlflow.log_metric("test_precision", prec_xgb)
    mlflow.log_metric("test_recall",    rec_xgb)
    mlflow.log_metric("test_f1",        f1_xgb)
    mlflow.log_metric("test_roc_auc",   auc_xgb)

    print("\n=== TUNED XGBOOST — Test Set ===")
    print(f"  Accuracy  : {acc_xgb*100:.2f}%")
    print(f"  ROC-AUC   : {auc_xgb:.4f}")
    print()
    print(classification_report(y_test, y_pred_xgb, target_names=["Denied(0)", "Approved(1)"], zero_division=0))

    mlflow.xgboost.log_model(best_xgb, artifact_path="model", input_example=X_test.iloc[:3])
    xgb_run_id = xgb_run.info.run_id
    print(f"  MLflow run ID: {xgb_run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — MLflow Model Serving (Databricks CE Pattern)
# MAGIC
# MAGIC Since **Databricks Community Edition** does not support the formal MLflow Model Registry (`mlflow.register_model`), we serve the best model directly using its `runs:/<run_id>/model` URI.

# COMMAND ----------

# Select the best run
if auc_xgb >= auc_lr:
    best_run_id   = xgb_run_id
    best_model_name = "XGBoost (Tuned)"
else:
    best_run_id   = lr_run_id
    best_model_name = "Logistic Regression"

# Construct URI to load directly from the Run artifact (CE compatible)
prod_model_uri = f"runs:/{best_run_id}/model"

print(f"Selected Best Model: {best_model_name}")
print(f"Serving Model URI  : {prod_model_uri}")

# Load the model back into memory to act as our "Production" serving endpoint
if best_model_name == "XGBoost (Tuned)":
    prod_model = mlflow.xgboost.load_model(prod_model_uri)
    def predict_fn(data): return prod_model.predict(data), prod_model.predict_proba(data)[:, 1]
else:
    prod_model = mlflow.sklearn.load_model(prod_model_uri)
    def predict_fn(data): return prod_model.predict(scaler.transform(data)), prod_model.predict_proba(scaler.transform(data))[:, 1]

print("✅ Model loaded and ready for inference!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Inference on New Sample Data

# COMMAND ----------

# Let's create some brand new "unseen" sample data
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
        "billing_ratio": 3.5,     # Grossly overbilled
        "cost_diff": 15000,       # Huge difference
        "high_cost_flag": 1,
        "provider_claim_count": 5,
        "provider_specialty_enc": 1,
        "severity_score": 1,      # Low severity
        "diag_claim_count": 20,
        "diagnosis_category_enc": 3,
        "is_billed_missing": 0,
        "is_proc_missing": 1,     # Missing procedure code! High risk
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

print("Predicting on 3 new sample claims...\n")

new_preds, new_probs = predict_fn(new_claims_data)

for i, (pred, prob) in enumerate(zip(new_preds, new_probs)):
    claim_type = "Normal Claim" if i == 0 else ("Highly Suspicious" if i == 1 else "Perfect Claim")
    deny_p = 1 - prob
    decision = "DENIED" if pred == 0 else "APPROVED"
    print(f"Sample {i+1} ({claim_type}):")
    print(f"  Probability of Denial: {deny_p*100:.1f}%")
    print(f"  Final Decision       : {decision}\n")
