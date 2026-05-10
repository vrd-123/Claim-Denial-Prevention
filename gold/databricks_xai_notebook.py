# Databricks notebook source
# MAGIC %md
# MAGIC # Step 11 — Explainable AI (XAI): SHAP-Based Claim Denial Explanations
# MAGIC
# MAGIC **Lineage:**
# MAGIC ```
# MAGIC workspace.gold.gold_claim_features  ──► [SHAP TreeExplainer] ──► workspace.gold.gold_claim_explanations
# MAGIC ```
# MAGIC
# MAGIC **What this notebook does:**
# MAGIC - Loads the best trained XGBoost model from MLflow using the run URI
# MAGIC - Computes per-claim SHAP values for every row in `gold_claim_features`
# MAGIC - Selects the **Top 3 most impactful features** per claim (by SHAP magnitude)
# MAGIC - Translates raw technical feature names into human-readable business reasons
# MAGIC - Writes the final explanation table to `workspace.gold.gold_claim_explanations`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1 — Install SHAP

# COMMAND ----------

# MAGIC %pip install "numpy<2" "scipy<1.14" "xgboost==3.2.0" "shap>=0.45"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2 — Imports & Config

# COMMAND ----------

# DBTITLE 1,Cell 5
import shap
import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# ── MLflow Config ──────────────────────────────────────────────────────────
# Note: On Databricks, the default tracking URI is automatically set to workspace tracking
# We just need to set the experiment name
EXPERIMENT_NAME = "/Users/varadnaik03@gmail.com/claim_denial_prevention"

# ── Constants ─────────────────────────────────────────────────────────────────
# Replace this with the actual XGBoost run_id from your training notebook
XGB_RUN_ID = "7f706dc17f7240deb385f9fb451c428e"

FEATURE_COLS = [
    "billing_ratio",
    "cost_diff",
    "high_cost_flag",
    "provider_claim_count",
    "provider_specialty_enc",
    "severity_score",
    "diag_claim_count",
    "diag_category_enc",
    "is_billed_missing",
    "is_proc_missing",
    "is_diag_missing",
    "claim_age_days",
]

TARGET_COL = "denial_flag"
ID_COL     = "claim_id"
TOP_N      = 3  # Capture the top 3 drivers per claim (see design rationale in docs)

print("=" * 60)
print("SHAP EXPLAINABILITY PIPELINE — CONFIG")
print("=" * 60)
print(f"SHAP version         : {shap.__version__}")
print(f"MLflow Tracking      : Databricks Workspace")
print(f"Experiment           : {EXPERIMENT_NAME}")
print(f"XGBoost Run ID       : {XGB_RUN_ID}")
print(f"Feature count        : {len(FEATURE_COLS)}")
print(f"Top-N Reasons        : {TOP_N}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3 — Load Gold Feature Table

# COMMAND ----------

feat_spark = spark.table("workspace.gold.gold_claim_features")
feat_pd    = feat_spark.toPandas()

X       = feat_pd[FEATURE_COLS]
y       = feat_pd[TARGET_COL]
ids     = feat_pd[ID_COL]

print(f"Loaded gold_claim_features: {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"Denied (0)  : {int((y==0).sum()):,}")
print(f"Approved (1): {int((y==1).sum()):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4 — Load XGBoost Model from MLflow & Predict

# COMMAND ----------

# DBTITLE 1,Cell 10
model_uri = f"runs:/{XGB_RUN_ID}/model"

print(f"Loading model from: {model_uri}")
print(f"Experiment: {EXPERIMENT_NAME}")

xgb_model = mlflow.xgboost.load_model(model_uri)
print("✅ Model loaded successfully!\n")

# Run inference on the full feature table 
y_prob_all = xgb_model.predict_proba(X)[:, 1]  # P(Approved)
y_pred_all = xgb_model.predict(X)               # 0=Denied, 1=Approved

print(f"Predictions generated for {len(y_pred_all):,} claims.")
print(f"  Predicted Denied  : {int((y_pred_all==0).sum()):,}")
print(f"  Predicted Approved: {int((y_pred_all==1).sum()):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5 — Compute SHAP Values (TreeExplainer)
# MAGIC
# MAGIC **Why TreeExplainer?**
# MAGIC - Specifically optimised for tree-based models (XGBoost, LightGBM, etc.)
# MAGIC - Runs in polynomial time using the tree structure — far faster than KernelSHAP.
# MAGIC - Produces exact SHAP values (not approximations), ensuring each explanation
# MAGIC   is mathematically consistent with the model's actual prediction.
# MAGIC
# MAGIC **Output shape:** `(n_claims, n_features)` — one SHAP value per feature per claim.
# MAGIC - A **negative** SHAP value pushes the prediction toward `0` (Denied).
# MAGIC - A **positive** SHAP value pushes the prediction toward `1` (Approved).

# COMMAND ----------

print("Initialising SHAP TreeExplainer...")

# Workaround for XGBoost 3.2.0 + SHAP 0.49.1 compatibility issue
# XGBoost 3.2.0 stores base_score as an array string '[value]', but SHAP expects a float
import json
booster = xgb_model.get_booster()
config = json.loads(booster.save_config())

# Extract base_score and convert from array string to float
base_score_str = config['learner']['learner_model_param']['base_score']
if base_score_str.startswith('[') and base_score_str.endswith(']'):
    # Parse array notation and extract first value
    base_score = float(json.loads(base_score_str)[0])
    config['learner']['learner_model_param']['base_score'] = str(base_score)
    booster.load_config(json.dumps(config))

# Now create TreeExplainer with the patched booster
explainer = shap.TreeExplainer(booster)

print(f"Computing SHAP values for {X.shape[0]:,} claims...")
shap_values = explainer.shap_values(X)  # shape: (n_claims, n_features)

print(f"✅ SHAP values computed. Matrix shape: {shap_values.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6 — Business Reason Mapping
# MAGIC
# MAGIC Maps each raw feature name to a pair of human-readable business explanations:
# MAGIC - **denial_reason**: Used when the feature is pushing the claim toward DENIED (negative SHAP).
# MAGIC - **approval_reason**: Used when the feature is pushing the claim toward APPROVED (positive SHAP).

# COMMAND ----------

# Each feature maps to a tuple: (denial_message, approval_message)
# The correct message is selected based on the sign of the SHAP value.
REASON_MAP = {
    "billing_ratio": (
        "Claim billed amount is significantly higher than the benchmark expected cost.",
        "Claim billed amount is within the accepted benchmark cost range.",
    ),
    "cost_diff": (
        "The absolute cost gap between billed and expected amounts is excessively large.",
        "The cost gap between billed and expected amounts is within acceptable limits.",
    ),
    "high_cost_flag": (
        "Claim has been flagged as an extreme high-cost outlier by the cost model.",
        "No high-cost outlier flag detected; billing appears standard.",
    ),
    "provider_claim_count": (
        "The provider's low historical claim volume indicates a higher operational risk profile.",
        "The provider has a high historical claim volume, suggesting a reliable submission pattern.",
    ),
    "provider_specialty_enc": (
        "The billing pattern is inconsistent with the provider's recorded medical specialty.",
        "The billing pattern is consistent with the provider's medical specialty.",
    ),
    "severity_score": (
        "The clinical severity level is inconsistent with the standard billing profile for this claim.",
        "The clinical severity level aligns with the expected billing profile.",
    ),
    "diag_claim_count": (
        "This diagnosis code has an unusually low historical claim frequency, indicating potential miscoding.",
        "This diagnosis code has a strong historical claim frequency, indicating a reliable submission.",
    ),
    "diagnosis_category_enc": (
        "The diagnosis category is inconsistent with the procedure and billing pattern submitted.",
        "The diagnosis category is consistent with the procedure and billing pattern.",
    ),
    "is_billed_missing": (
        "The claim billed amount was missing from the original source submission.",
        "The claim billed amount is present and valid in the original submission.",
    ),
    "is_proc_missing": (
        "The medical procedure code is missing or invalid in the claim submission.",
        "The medical procedure code is present and valid.",
    ),
    "is_diag_missing": (
        "The medical diagnosis code is missing or invalid in the claim submission.",
        "The medical diagnosis code is present and valid.",
    ),
    "claim_age_days": (
        "The claim was submitted significantly late relative to the service date.",
        "The claim was submitted promptly relative to the service date.",
    ),
}

def get_reason_text(feature_name: str, shap_value: float) -> str:
    """
    Returns the correct business reason message for a given feature,
    based on whether its SHAP value is pushing toward Denial (negative)
    or Approval (positive).
    """
    denial_msg, approval_msg = REASON_MAP.get(
        feature_name,
        (f"Feature '{feature_name}' influenced the denial decision.",
         f"Feature '{feature_name}' supported the approval decision.")
    )
    return denial_msg if shap_value < 0 else approval_msg

print("✅ Business reason mapping defined.")
print(f"   Coverage: {len(REASON_MAP)}/{len(FEATURE_COLS)} features mapped.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7 — Build Explanation Rows (Top 3 per Claim)
# MAGIC
# MAGIC For each claim:
# MAGIC 1. Identify the predicted status (DENIED / APPROVED).
# MAGIC 2. Sort all 12 feature SHAP values by **absolute magnitude** (largest impact first).
# MAGIC 3. Select the Top 3 most impactful features.
# MAGIC 4. Translate each feature to its business reason text using the sign of its SHAP value.

# COMMAND ----------

PROCESSED_AT = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

rows = []

for i in range(len(X)):
    claim_id         = ids.iloc[i]
    deny_probability = round(float(1 - y_prob_all[i]), 4)    # P(denial)
    predicted_status = "DENIED" if y_pred_all[i] == 0 else "APPROVED"

    # Get all 12 SHAP values for this claim
    claim_shap = shap_values[i]  # array of shape (12,)

    # Sort feature indices by absolute SHAP magnitude (descending — highest impact first)
    sorted_indices = np.argsort(np.abs(claim_shap))[::-1]

    # Pick the Top 3
    top3 = sorted_indices[:TOP_N]

    row = {
        "claim_id"          : claim_id,
        "denial_probability": deny_probability,
        "predicted_status"  : predicted_status,
        "processed_at"      : PROCESSED_AT,
    }

    # Populate reason_1, reason_2, reason_3 columns
    for rank, feat_idx in enumerate(top3, start=1):
        feat_name  = FEATURE_COLS[feat_idx]
        shap_val   = round(float(claim_shap[feat_idx]), 4)
        reason_txt = get_reason_text(feat_name, shap_val)

        row[f"reason_{rank}_feature"] = feat_name
        row[f"reason_{rank}_text"]    = reason_txt
        row[f"reason_{rank}_impact"]  = shap_val

    rows.append(row)

explanations_pd = pd.DataFrame(rows)

# Ensure consistent column ordering
col_order = [
    "claim_id", "denial_probability", "predicted_status",
    "reason_1_feature", "reason_1_text", "reason_1_impact",
    "reason_2_feature", "reason_2_text", "reason_2_impact",
    "reason_3_feature", "reason_3_text", "reason_3_impact",
    "processed_at",
]
explanations_pd = explanations_pd[col_order]

print("=" * 60)
print("EXPLANATION TABLE — BUILD COMPLETE")
print("=" * 60)
print(f"Total rows          : {len(explanations_pd):,}")
print(f"Predicted Denied    : {(explanations_pd['predicted_status']=='DENIED').sum():,}")
print(f"Predicted Approved  : {(explanations_pd['predicted_status']=='APPROVED').sum():,}")
print(f"Columns             : {list(explanations_pd.columns)}")
print("\nSample (first 3 rows):")
print(explanations_pd[["claim_id", "predicted_status", "denial_probability",
                        "reason_1_feature", "reason_1_text"]].head(3).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8 — Write `gold_claim_explanations` to Databricks Gold Layer

# COMMAND ----------

# Convert to Spark DataFrame for Delta write
explanations_spark = spark.createDataFrame(explanations_pd)

print("Schema of gold_claim_explanations:")
explanations_spark.printSchema()

# Write to Gold layer as a managed Delta table
(
    explanations_spark
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("workspace.gold.gold_claim_explanations")
)

print("\n✅ Table written: workspace.gold.gold_claim_explanations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9 — Verify: Read Back & Spot-Check

# COMMAND ----------

verify_df = spark.table("workspace.gold.gold_claim_explanations")

print("=" * 60)
print("VERIFICATION: workspace.gold.gold_claim_explanations")
print("=" * 60)
print(f"Row count : {verify_df.count():,}")

# Show a sample of DENIED claims
print("\n--- Sample DENIED Claims ---")
(
    verify_df
    .filter("predicted_status = 'DENIED'")
    .select(
        "claim_id", "denial_probability",
        "reason_1_feature", "reason_1_text", "reason_1_impact",
        "reason_2_feature", "reason_2_text", "reason_2_impact",
        "reason_3_feature", "reason_3_text", "reason_3_impact",
    )
    .orderBy("denial_probability", ascending=False)
    .limit(5)
    .show(truncate=60)
)

# Show a sample of APPROVED claims
print("--- Sample APPROVED Claims ---")
(
    verify_df
    .filter("predicted_status = 'APPROVED'")
    .select(
        "claim_id", "denial_probability",
        "reason_1_feature", "reason_1_text",
        "reason_2_feature", "reason_2_text",
    )
    .orderBy("denial_probability")
    .limit(5)
    .show(truncate=60)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10 — Top Denial Reasons Summary (Aggregate View)
# MAGIC
# MAGIC Aggregates across all DENIED claims to show which features are
# MAGIC most frequently the *primary* driver of denials. Useful for
# MAGIC identifying systemic issues in your claims submission process.

# COMMAND ----------

denied_pd = explanations_pd[explanations_pd["predicted_status"] == "DENIED"].copy()

# Count how often each feature appears as the #1 reason across all denied claims
top_reasons_summary = (
    denied_pd["reason_1_feature"]
    .value_counts()
    .reset_index()
)
top_reasons_summary.columns = ["primary_denial_feature", "count"]
top_reasons_summary["pct_of_denied_claims"] = (
    top_reasons_summary["count"] / len(denied_pd) * 100
).round(1)

print("=" * 60)
print("TOP PRIMARY DENIAL DRIVERS ACROSS ALL DENIED CLAIMS")
print("=" * 60)
print(top_reasons_summary.to_string(index=False))
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11 — Pipeline Summary

# COMMAND ----------

print("=" * 65)
print("  WEEK 5 — EXPLAINABLE AI PIPELINE SUMMARY")
print("=" * 65)
print(f"  Input table    : workspace.gold.gold_claim_features")
print(f"  Output table   : workspace.gold.gold_claim_explanations")
print(f"  Total claims   : {len(explanations_pd):,}")
print(f"  XAI method     : SHAP TreeExplainer (exact, not approximate)")
print(f"  Model used     : XGBoost (Tuned) — run ID: {XGB_RUN_ID}")
print(f"  Top-N reasons  : {TOP_N} per claim")
print()
print(f"  Output columns:")
print(f"    claim_id            — Primary key")
print(f"    denial_probability  — P(denial) score from model")
print(f"    predicted_status    — DENIED / APPROVED")
print(f"    reason_1_feature    — Top driver (technical name)")
print(f"    reason_1_text       — Top driver (business explanation)")
print(f"    reason_1_impact     — SHAP value (negative = denial driver)")
print(f"    reason_2/3_*        — Same columns for 2nd and 3rd drivers")
print(f"    processed_at        — Pipeline run timestamp")
print("=" * 65)
