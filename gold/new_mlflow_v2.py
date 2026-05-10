# Databricks notebook source
# MAGIC %md
# MAGIC # Step 10B — Model Training, Hyperparameter Tuning & MLflow Tracking
# MAGIC
# MAGIC ## Databricks Community Edition — Production-Safe Version
# MAGIC
# MAGIC This notebook:
# MAGIC - Trains **Logistic Regression**, **Random Forest**, and **XGBoost** (tuned)
# MAGIC - Uses cross-validation with an overfitting gap diagnostic
# MAGIC - Logs all metrics, params, artifacts, and models to **Databricks Managed MLflow**
# MAGIC - Generates feature importance plots and confusion matrices as MLflow artifacts
# MAGIC - Selects the best model by ROC-AUC and runs inference on new claims
# MAGIC - Is fully safe to run on Databricks Community Edition via Run All

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Imports & MLflow Configuration

# COMMAND ----------

pip install xgboost

# COMMAND ----------

import os
import warnings
import tempfile

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — required on Databricks CE
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Set registry URI before any operations that might trigger auto-detection
mlflow.set_registry_uri("databricks")

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    make_scorer,
)

from xgboost import XGBClassifier

# ── GLOBAL CONFIG ────────────────────────────────────────────────────────────

RANDOM_SEED = 42
TRAIN_RATIO = 0.80
N_FOLDS     = 5

# ── FEATURE / TARGET COLUMNS (must match workspace.gold.gold_claim_features) ─
# NOTE: The feature engineering notebook writes "diag_category_enc".
#       This list uses that exact name — DO NOT change to "diagnosis_category_enc".

FEATURE_COLS = [
    "billing_ratio",
    "cost_diff",
    "high_cost_flag",
    "provider_claim_count",
    "provider_specialty_enc",
    "severity_score",
    "diag_claim_count",
    "diag_category_enc",        # ← corrected from "diagnosis_category_enc"
    "is_billed_missing",
    "is_proc_missing",
    "is_diag_missing",
    "claim_age_days",
]

TARGET_COL = "denial_flag"
ID_COL     = "claim_id"

# ── MLflow: use Databricks managed tracking (shows in the Experiments UI tab) ─
# On Databricks CE, NOT setting a file:// URI means MLflow automatically uses
# the cluster's built-in managed tracking server.  This makes runs visible in
# the "Experiments" sidebar of your notebook without any extra setup.

EXPERIMENT_NAME = "/Users/varadnaik03@gmail.com/claim_denial_prevention"

# End any run that may have been left open by a previous crashed execution
if mlflow.active_run():
    mlflow.end_run()

mlflow.set_experiment(EXPERIMENT_NAME)

print("=" * 65)
print("MLFLOW CONFIGURATION")
print("=" * 65)
print(f"Tracking      : Databricks Managed MLflow (built-in)")
print(f"Experiment    : {EXPERIMENT_NAME}")
print(f"  → Open 'Experiments' in the left sidebar to view runs")
print(f"Train/Test    : {int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}")
print(f"CV Folds      : {N_FOLDS}")
print("=" * 65)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Load Gold Feature Table

# COMMAND ----------

# ── Load from the correct catalog path ───────────────────────────────────────
# The feature engineering notebook writes to workspace.gold.gold_claim_features
# (NOT workspace.default.gold_claim_features — that path would raise an error)

feat_spark = spark.table("workspace.gold.gold_claim_features")
feat_pd    = feat_spark.toPandas()

# Validate that all expected columns are present before proceeding
missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in feat_pd.columns]
if missing_cols:
    raise ValueError(
        f"The following columns are missing from gold_claim_features: {missing_cols}\n"
        f"Available columns: {list(feat_pd.columns)}"
    )

X = feat_pd[FEATURE_COLS].copy()
y = feat_pd[TARGET_COL].copy()

denied   = int((y == 0).sum())
approved = int((y == 1).sum())
total    = len(y)

print("=" * 65)
print("DATASET SUMMARY")
print("=" * 65)
print(f"Dataset Shape    : {X.shape}")
print(f"Total Claims     : {total:,}")
print(f"Denied  (label=0): {denied:,}  ({denied/total*100:.1f}%)")
print(f"Approved(label=1): {approved:,}  ({approved/total*100:.1f}%)")
print(f"Class Imbalance  : {max(denied,approved)/min(denied,approved):.2f}x")
print("=" * 65)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Train / Test Split + Scaling

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size     = 1 - TRAIN_RATIO,
    stratify      = y,
    random_state  = RANDOM_SEED,
)

print("=" * 65)
print("TRAIN / TEST SPLIT")
print("=" * 65)
print(
    f"Train : {len(X_train):,} rows | "
    f"Denied={(y_train==0).sum():,}  Approved={(y_train==1).sum():,}"
)
print(
    f"Test  : {len(X_test):,} rows  | "
    f"Denied={(y_test==0).sum():,}  Approved={(y_test==1).sum():,}"
)
print("=" * 65)

# StandardScaler for Logistic Regression only
# Tree-based models (RF, XGB) will use the unscaled arrays directly
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

# ── Shared CV strategy (reused across all models) ────────────────────────────
cv_strategy = StratifiedKFold(
    n_splits     = N_FOLDS,
    shuffle      = True,
    random_state = RANDOM_SEED,
)

# ── Shared scorer: F1 on the denied class (label=0) ─────────────────────────
f1_denied_scorer = make_scorer(f1_score, zero_division=0, pos_label=0)

# ── Helper: compute all test metrics in one call ─────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "test_accuracy" : accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred, zero_division=0, pos_label=0),
        "test_recall"   : recall_score   (y_true, y_pred, zero_division=0, pos_label=0),
        "test_f1"       : f1_score       (y_true, y_pred, zero_division=0, pos_label=0),
        "test_roc_auc"  : roc_auc_score  (y_true, y_prob),
    }

# ── Helper: save confusion matrix as a PNG and return the file path ───────────
def save_confusion_matrix(y_true, y_pred, model_name, tmp_dir):
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Denied(0)", "Approved(1)"])
    ax.set_yticklabels(["Denied(0)", "Approved(1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(tmp_dir, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path

# ── Helper: save feature importance bar chart and return file path ────────────
def save_feature_importance(importances, feature_names, model_name, tmp_dir):
    idx  = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(importances)), importances[idx], color="#4C72B0")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}")
    plt.tight_layout()
    path = os.path.join(tmp_dir, f"feature_importance_{model_name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path

# ── Helper: print a formatted results block ──────────────────────────────────
def print_results(model_name, metrics, cv_score, run_id):
    gap = cv_score - metrics["test_f1"]          # positive → generalising well
    print("=" * 65)
    print(f"  {model_name}")
    print("=" * 65)
    print(f"  Accuracy       : {metrics['test_accuracy']*100:.2f}%")
    print(f"  Precision      : {metrics['test_precision']:.4f}")
    print(f"  Recall         : {metrics['test_recall']:.4f}")
    print(f"  F1 (denied)    : {metrics['test_f1']:.4f}")
    print(f"  ROC-AUC        : {metrics['test_roc_auc']:.4f}")
    print(f"  CV F1 (mean)   : {cv_score:.4f}")
    print(f"  CV–Test gap    : {gap:+.4f}  {'✅ healthy' if abs(gap) < 0.05 else '⚠️  check for overfit/underfit'}")
    print(f"  Run ID         : {run_id}")
    print("=" * 65)

# Storage for cross-run comparison
run_summary = {}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Model A: Logistic Regression (Baseline)
# MAGIC
# MAGIC A regularised linear baseline.  We tune `C` via cross-validation
# MAGIC and log the CV score alongside the test score so you can spot
# MAGIC overfitting directly in the MLflow Experiments UI.

# COMMAND ----------

# End any orphaned run before starting a new one
if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_name="logistic_regression") as lr_run:

    lr_params = {
        "max_iter"    : 1000,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
        "solver"      : "lbfgs",
        "C"           : 1.0,          # L2 regularisation strength
    }
    mlflow.log_params(lr_params)
    mlflow.log_param("model_type", "LogisticRegression")

    lr = LogisticRegression(**lr_params)
    lr.fit(X_train_sc, y_train)

    # Cross-validation score (on training data only — no test leakage)
    cv_scores_lr = cross_val_score(
        lr, X_train_sc, y_train,
        cv=cv_strategy, scoring=f1_denied_scorer,
    )
    cv_mean_lr = float(cv_scores_lr.mean())
    cv_std_lr  = float(cv_scores_lr.std())

    y_pred_lr = lr.predict(X_test_sc)
    y_prob_lr = lr.predict_proba(X_test_sc)[:, 1]
    metrics_lr = compute_metrics(y_test, y_pred_lr, y_prob_lr)

    # Log all metrics
    mlflow.log_metrics(metrics_lr)
    mlflow.log_metric("cv_f1_mean",     cv_mean_lr)
    mlflow.log_metric("cv_f1_std",      cv_std_lr)
    mlflow.log_metric("overfit_gap",    cv_mean_lr - metrics_lr["test_f1"])

    # Log classification report as a text artifact
    report = classification_report(
        y_test, y_pred_lr,
        target_names=["Denied(0)", "Approved(1)"],
        zero_division=0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        # Confusion matrix artifact
        cm_path = save_confusion_matrix(y_test, y_pred_lr, "Logistic_Regression", tmp)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # Classification report artifact
        rpt_path = os.path.join(tmp, "classification_report.txt")
        with open(rpt_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(rpt_path, artifact_path="reports")

        # Log model
        mlflow.sklearn.log_model(
            lr,
            artifact_path="model",
            input_example=X_test[FEATURE_COLS].iloc[:3],
        )

    lr_run_id = lr_run.info.run_id
    run_summary["Logistic Regression"] = {
        "run_id": lr_run_id,
        "metrics": metrics_lr,
        "cv_f1": cv_mean_lr,
    }

print_results("Logistic Regression", metrics_lr, cv_mean_lr, lr_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Model B: Random Forest (with Randomized Search CV)
# MAGIC
# MAGIC Random Forest adds ensemble-level regularisation through
# MAGIC `max_features`, `min_samples_leaf`, and `max_depth`.
# MAGIC These hyperparameters are tuned via RandomizedSearchCV.

# COMMAND ----------

if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_name="random_forest_tuned") as rf_run:

    scale_pos = approved / denied     # same imbalance correction as XGBoost

    rf_param_grid = {
        "n_estimators"     : [200, 300, 400],
        "max_depth"        : [6, 8, 10, None],    # None = fully grown (capped by min_samples_leaf)
        "min_samples_leaf" : [2, 4, 8],           # higher value = more regularisation
        "max_features"     : ["sqrt", "log2"],    # limits features per split → reduces variance
        "class_weight"     : ["balanced"],
    }

    base_rf = RandomForestClassifier(
        random_state=RANDOM_SEED,
        n_jobs=1,              # n_jobs=1 avoids multiprocessing issues on Databricks CE
    )

    rf_search = RandomizedSearchCV(
        estimator          = base_rf,
        param_distributions= rf_param_grid,
        n_iter             = 20,
        scoring            = f1_denied_scorer,
        cv                 = cv_strategy,
        verbose            = 1,
        n_jobs             = 1,          # safe for Databricks CE
        random_state       = RANDOM_SEED,
        refit              = True,
        return_train_score = True,       # lets us detect overfitting from CV results
    )

    print("=" * 65)
    print("RUNNING RANDOMIZED SEARCH — RANDOM FOREST")
    print("=" * 65)
    rf_search.fit(X_train, y_train)

    print(f"\nBest Params : {rf_search.best_params_}")
    print(f"Best CV F1  : {rf_search.best_score_:.4f}")

    best_rf    = rf_search.best_estimator_
    cv_mean_rf = float(rf_search.best_score_)

    mlflow.log_params(rf_search.best_params_)
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_iter_search", 20)

    y_pred_rf = best_rf.predict(X_test)
    y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
    metrics_rf = compute_metrics(y_test, y_pred_rf, y_prob_rf)

    mlflow.log_metrics(metrics_rf)
    mlflow.log_metric("cv_f1_mean",  cv_mean_rf)
    mlflow.log_metric("overfit_gap", cv_mean_rf - metrics_rf["test_f1"])

    report_rf = classification_report(
        y_test, y_pred_rf,
        target_names=["Denied(0)", "Approved(1)"],
        zero_division=0,
    )

    with tempfile.TemporaryDirectory() as tmp:
        cm_path = save_confusion_matrix(y_test, y_pred_rf, "Random_Forest", tmp)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        fi_path = save_feature_importance(
            best_rf.feature_importances_, FEATURE_COLS, "Random_Forest", tmp
        )
        mlflow.log_artifact(fi_path, artifact_path="plots")

        rpt_path = os.path.join(tmp, "classification_report.txt")
        with open(rpt_path, "w") as f:
            f.write(report_rf)
        mlflow.log_artifact(rpt_path, artifact_path="reports")

        mlflow.sklearn.log_model(
            best_rf,
            artifact_path="model",
            input_example=X_test[FEATURE_COLS].iloc[:3],
        )

    rf_run_id = rf_run.info.run_id
    run_summary["Random Forest (Tuned)"] = {
        "run_id": rf_run_id,
        "metrics": metrics_rf,
        "cv_f1": cv_mean_rf,
    }

print_results("Random Forest (Tuned)", metrics_rf, cv_mean_rf, rf_run_id)
print("\nClassification Report:\n")
print(report_rf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Model C: XGBoost (with Randomized Search CV + Early Stopping)
# MAGIC
# MAGIC XGBoost is tuned with a wider regularisation search space.
# MAGIC `early_stopping_rounds` halts training when the validation
# MAGIC log-loss stops improving — this directly prevents overfitting
# MAGIC without relying only on hyperparameter search.

# COMMAND ----------

if mlflow.active_run():
    mlflow.end_run()

with mlflow.start_run(run_name="xgboost_tuned") as xgb_run:

    scale_pos = approved / denied

    # Wider regularisation search space vs the original
    xgb_param_grid = {
        "n_estimators"    : [200, 300, 400, 500],
        "max_depth"       : [3, 4, 5, 6],
        "learning_rate"   : [0.01, 0.03, 0.05, 0.1],
        "subsample"       : [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma"           : [0, 0.1, 0.2, 0.3, 0.5],   # min loss reduction to split
        "reg_alpha"       : [0, 0.05, 0.1, 0.3, 0.5],  # L1 regularisation
        "reg_lambda"      : [1.0, 1.5, 2.0, 3.0],      # L2 regularisation
    }

    # NOTE: early_stopping_rounds is NOT passed to the constructor here because
    # RandomizedSearchCV manages fit() calls internally.  We apply early stopping
    # in a second dedicated fit after the best params are found (see below).
    base_xgb = XGBClassifier(
        eval_metric      = "logloss",
        random_state     = RANDOM_SEED,
        tree_method      = "hist",
        scale_pos_weight = scale_pos,
        use_label_encoder= False,
    )

    xgb_search = RandomizedSearchCV(
        estimator          = base_xgb,
        param_distributions= xgb_param_grid,
        n_iter             = 30,          # more iterations than original (was 20)
        scoring            = f1_denied_scorer,
        cv                 = cv_strategy,
        verbose            = 1,
        n_jobs             = 1,           # safe for Databricks CE
        random_state       = RANDOM_SEED,
        refit              = False,        # we will refit manually with early stopping
        return_train_score = True,
    )

    print("=" * 65)
    print("RUNNING RANDOMIZED SEARCH — XGBOOST")
    print("=" * 65)
    xgb_search.fit(X_train, y_train)

    best_params_xgb = xgb_search.best_params_
    cv_mean_xgb     = float(xgb_search.best_score_)

    print(f"\nBest Params : {best_params_xgb}")
    print(f"Best CV F1  : {cv_mean_xgb:.4f}")

    # ── Re-fit best model WITH early stopping to avoid over-training ──────────
    # Split a small internal validation set from the training data for early stopping
    X_tr_es, X_val_es, y_tr_es, y_val_es = train_test_split(
        X_train, y_train,
        test_size    = 0.10,
        stratify     = y_train,
        random_state = RANDOM_SEED,
    )

    final_xgb = XGBClassifier(
        **best_params_xgb,
        eval_metric          = "logloss",
        random_state         = RANDOM_SEED,
        tree_method          = "hist",
        scale_pos_weight     = scale_pos,
        use_label_encoder    = False,
        early_stopping_rounds= 20,         # stops if no improvement for 20 rounds
    )

    final_xgb.fit(
        X_tr_es, y_tr_es,
        eval_set            = [(X_val_es, y_val_es)],
        verbose             = False,
    )

    actual_n_trees = final_xgb.best_iteration + 1
    print(f"\nEarly stopping: used {actual_n_trees} trees "
          f"(out of max {best_params_xgb.get('n_estimators', '?')})")

    mlflow.log_params(best_params_xgb)
    mlflow.log_param("model_type",          "XGBoost")
    mlflow.log_param("early_stopping_used", True)
    mlflow.log_param("actual_n_trees",      actual_n_trees)
    mlflow.log_param("n_iter_search",       30)

    y_pred_xgb = final_xgb.predict(X_test)
    y_prob_xgb = final_xgb.predict_proba(X_test)[:, 1]
    metrics_xgb = compute_metrics(y_test, y_pred_xgb, y_prob_xgb)

    mlflow.log_metrics(metrics_xgb)
    mlflow.log_metric("cv_f1_mean",  cv_mean_xgb)
    mlflow.log_metric("overfit_gap", cv_mean_xgb - metrics_xgb["test_f1"])

    report_xgb = classification_report(
        y_test, y_pred_xgb,
        target_names=["Denied(0)", "Approved(1)"],
        zero_division=0,
    )

    with tempfile.TemporaryDirectory() as tmp:
        cm_path = save_confusion_matrix(y_test, y_pred_xgb, "XGBoost", tmp)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        fi_path = save_feature_importance(
            final_xgb.feature_importances_, FEATURE_COLS, "XGBoost", tmp
        )
        mlflow.log_artifact(fi_path, artifact_path="plots")

        rpt_path = os.path.join(tmp, "classification_report.txt")
        with open(rpt_path, "w") as f:
            f.write(report_xgb)
        mlflow.log_artifact(rpt_path, artifact_path="reports")

        mlflow.xgboost.log_model(
            final_xgb,
            artifact_path="model",
            input_example=X_test[FEATURE_COLS].iloc[:3],
        )

    xgb_run_id = xgb_run.info.run_id
    run_summary["XGBoost (Tuned)"] = {
        "run_id": xgb_run_id,
        "metrics": metrics_xgb,
        "cv_f1": cv_mean_xgb,
    }

print_results("XGBoost (Tuned)", metrics_xgb, cv_mean_xgb, xgb_run_id)
print("\nClassification Report:\n")
print(report_xgb)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Model Comparison & Best Model Selection
# MAGIC
# MAGIC Compares all three models side-by-side and selects the one with
# MAGIC the highest test ROC-AUC.  The comparison table is also logged
# MAGIC as an MLflow artifact so it appears in the Experiments UI.

# COMMAND ----------

if mlflow.active_run():
    mlflow.end_run() 

print("=" * 65)
print("MODEL COMPARISON")
print("=" * 65)
print(f"{'Model':<28} {'ROC-AUC':>8} {'F1':>8} {'Accuracy':>9} {'CV F1':>8} {'Gap':>7}")
print("-" * 65)

best_auc        = -1.0
best_model_name = None

for name, info in run_summary.items():
    m   = info["metrics"]
    gap = info["cv_f1"] - m["test_f1"]
    print(
        f"{name:<28} {m['test_roc_auc']:>8.4f} {m['test_f1']:>8.4f} "
        f"{m['test_accuracy']*100:>8.2f}% {info['cv_f1']:>8.4f} {gap:>+7.4f}"
    )
    if m["test_roc_auc"] > best_auc:
        best_auc        = m["test_roc_auc"]
        best_model_name = name

print("-" * 65)
print(f"\n🏆  Best Model : {best_model_name}  (ROC-AUC = {best_auc:.4f})")
print("=" * 65)

best_run_id    = run_summary[best_model_name]["run_id"]
prod_model_uri = f"runs:/{best_run_id}/model"

# ── Log the comparison table as an artifact on the best run ──────────────────
with mlflow.start_run(run_id=best_run_id):
    with tempfile.TemporaryDirectory() as tmp:
        cmp_rows = []
        for name, info in run_summary.items():
            m = info["metrics"]
            cmp_rows.append({
                "model"       : name,
                "roc_auc"     : round(m["test_roc_auc"], 4),
                "f1"          : round(m["test_f1"], 4),
                "precision"   : round(m["test_precision"], 4),
                "recall"      : round(m["test_recall"], 4),
                "accuracy"    : round(m["test_accuracy"], 4),
                "cv_f1"       : round(info["cv_f1"], 4),
                "overfit_gap" : round(info["cv_f1"] - m["test_f1"], 4),
                "run_id"      : info["run_id"],
                "selected"    : (name == best_model_name),
            }) 
        cmp_df   = pd.DataFrame(cmp_rows)
        cmp_path = os.path.join(tmp, "model_comparison.csv")
        cmp_df.to_csv(cmp_path, index=False)
        mlflow.log_artifact(cmp_path, artifact_path="reports")
        mlflow.log_param("final_selected_model", best_model_name)
        mlflow.log_metric("final_roc_auc", best_auc)

print(f"\nModel URI   : {prod_model_uri}")
print("The comparison table has been saved to the best run's artifacts.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 — Load Production Model & Inference on New Claims

# COMMAND ----------

# ── Load the best model using its runs:/ URI ─────────────────────────────────
# mlflow.pyfunc.load_model works for both sklearn and xgboost flavours
prod_model = mlflow.pyfunc.load_model(prod_model_uri)

def predict_fn(data: pd.DataFrame):
    """
    Returns (predictions, denial_probabilities).

    For the pyfunc flavour, predict() returns class labels (as floats).
    We reconstruct denial probability as 1 – P(approved) by falling back
    to the native model when predict_proba is needed.
    """
    # Use native model flavour for predict_proba
    if best_model_name == "XGBoost (Tuned)":
        native = mlflow.xgboost.load_model(prod_model_uri)
        preds  = native.predict(data)
        probs  = native.predict_proba(data)[:, 1]
    else:
        native = mlflow.sklearn.load_model(prod_model_uri)
        if "Logistic" in best_model_name:
            data_in = scaler.transform(data)
        else:
            data_in = data
        preds = native.predict(data_in)
        probs = native.predict_proba(data_in)[:, 1]

    return preds, probs

print("✅ Production model loaded successfully!")
print(f"   Model        : {best_model_name}")
print(f"   Run URI      : {prod_model_uri}")

# ── Sample claims for inference ───────────────────────────────────────────────
new_claims_data = pd.DataFrame([
    {   # Low-risk claim: billing near expected, established provider
        "billing_ratio": 1.1, "cost_diff": 500, "high_cost_flag": 0,
        "provider_claim_count": 45, "provider_specialty_enc": 2,
        "severity_score": 2, "diag_claim_count": 150,
        "diag_category_enc": 1, "is_billed_missing": 0,
        "is_proc_missing": 0, "is_diag_missing": 0, "claim_age_days": 10,
    },
    {   # High-risk claim: severe overbilling, new provider, missing procedure
        "billing_ratio": 3.5, "cost_diff": 15000, "high_cost_flag": 1,
        "provider_claim_count": 5, "provider_specialty_enc": 1,
        "severity_score": 1, "diag_claim_count": 20,
        "diag_category_enc": 3, "is_billed_missing": 1,
        "is_proc_missing": 1, "is_diag_missing": 0, "claim_age_days": 45,
    },
    {   # Approved claim: under-billed, high-volume established provider
        "billing_ratio": 0.9, "cost_diff": -100, "high_cost_flag": 0,
        "provider_claim_count": 200, "provider_specialty_enc": 0,
        "severity_score": 3, "diag_claim_count": 500,
        "diag_category_enc": 2, "is_billed_missing": 0,
        "is_proc_missing": 0, "is_diag_missing": 0, "claim_age_days": 2,
    },
])

print("\n" + "=" * 65)
print("INFERENCE ON NEW CLAIMS")
print("=" * 65)

new_preds, new_probs = predict_fn(new_claims_data)

for i, (pred, prob) in enumerate(zip(new_preds, new_probs)):
    deny_prob  = 1.0 - prob
    decision   = "DENIED ❌" if pred == 0 else "APPROVED ✅"
    risk_level = "HIGH" if deny_prob > 0.6 else ("MEDIUM" if deny_prob > 0.35 else "LOW")
    print(f"\n  Sample Claim {i+1}")
    print(f"  {'─'*40}")
    print(f"  Denial Probability : {deny_prob*100:.2f}%")
    print(f"  Risk Level         : {risk_level}")
    print(f"  Final Decision     : {decision}")

print("\n✅ Inference completed successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9 — How to View Results in MLflow UI
# MAGIC
# MAGIC > **You do not need a separate URL.**
# MAGIC > Databricks Community Edition has MLflow built in.
# MAGIC >
# MAGIC > ### Steps to open the Experiments UI:
# MAGIC > 1. Click **"Experiments"** in the left sidebar of your Databricks workspace
# MAGIC > 2. Find the experiment named **`claim_denial_prevention`** under your user path
# MAGIC > 3. Click it to see all runs side-by-side
# MAGIC > 4. Use the **"Columns"** button to add `cv_f1_mean`, `overfit_gap`, `test_roc_auc`
# MAGIC > 5. Click any run to see its artifacts (confusion matrix, feature importance plots, classification report)
# MAGIC >
# MAGIC > ### What you can compare in the UI:
# MAGIC > | Metric | What it tells you |
# MAGIC > |---|---|
# MAGIC > | `test_roc_auc` | Overall discrimination power |
# MAGIC > | `test_f1` | F1 on the denied class |
# MAGIC > | `cv_f1_mean` | Average F1 across 5 folds |
# MAGIC > | `overfit_gap` | CV F1 − Test F1: small positive = healthy |
# MAGIC > | `test_precision` / `test_recall` | Precision-recall tradeoff |

# COMMAND ----------

print("=" * 65)
print("HOW TO VIEW RESULTS IN MLFLOW")
print("=" * 65) 
print()
print("  ▶  In your Databricks notebook:")
print("     Left sidebar → 'Experiments'")
print(f"     → Experiment: claim_denial_prevention")
print("     → You will see one row per model run")
print()
print("  ▶  Logged per run:")
print("     • Metrics  : test_roc_auc, test_f1, test_precision,")
print("                  test_recall, test_accuracy, cv_f1_mean, overfit_gap")
print("     • Params   : all hyperparameters used")
print("     • Artifacts: confusion_matrix.png, feature_importance.png,")
print("                  classification_report.txt, model_comparison.csv")
print("     • Model    : loadable via mlflow.pyfunc.load_model(uri)")
print()
print("=" * 65)
print("PRODUCTION MODEL SUMMARY")
print("=" * 65)
print(f"  Selected : {best_model_name}")
print(f"  ROC-AUC  : {best_auc:.4f}")
print(f"  Run URI  : {prod_model_uri}")
print()
print("  To reload this model in any notebook:")
print(f"  model = mlflow.pyfunc.load_model('{prod_model_uri}')")
print("=" * 65)
