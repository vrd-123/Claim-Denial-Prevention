# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC Reads from silver.silver_claims (already has pre_risk_score + boolean flags).
# MAGIC Joins silver.silver_cost on **proc_id`** to get expected_cost.
# MAGIC Computes billing_ratio, adds overbilling points → final_risk_score, then applies label.
# MAGIC
# MAGIC **Three columns added:**
# MAGIC
# MAGIC | Column | Description |
# MAGIC |---|---|
# MAGIC | billing_ratio | billed_amount / expected_cost (NULL when proc_id = UNKNOWN) |
# MAGIC | final_risk_score | pre_risk_score + overbilling points |
# MAGIC | denial_flag | **0 = Denied, 1 = Approved** |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports & Config

# COMMAND ----------

from pyspark.sql import functions as F

RANDOM_SEED    = 42    
DENY_THRESHOLD = 5      # final_risk_score >= 5 → Denied (0)
NOISE_RATE     = 0.02  #2% symmetric flip 

print(f"Threshold={DENY_THRESHOLD} | Noise={NOISE_RATE*100}% | Seed={RANDOM_SEED}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Silver Tables
# MAGIC
# MAGIC > **Note:** Silver tables are in the default schema (as written by databricks_silver_notebook.py).
# MAGIC > Cost table join key is procedure_code in silver_cost and proc_id in silver_claims.

# COMMAND ----------

# Load silver_claims (already has pre_risk_score and boolean flags from silver notebook)
claims = spark.table("workspace.silver.silver_claims")

# Load only what we need from silver_cost
cost = spark.table("workspace.silver.silver_cost").select("procedure_code", "expected_cost")

print(f"silver_claims rows : {claims.count():,}")
print(f"silver_cost rows   : {cost.count():,}")

# Quick schema check — confirm key columns exist
print("\nsilver_claims columns:", claims.columns)
print("silver_cost columns  :", cost.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Diagnostics: Check Join Key Overlap
# MAGIC
# MAGIC silver_claims.procedure_code joins to silver_cost.procedure_code.
# MAGIC Claims where procedure_code = 'UNKNOWN' will NOT match → expected_cost = NULL (correct behaviour).

# COMMAND ----------

# How many claims have a real procedure_code vs UNKNOWN?
print("procedure_code distribution in silver_claims:")
claims.groupBy(
    F.when(F.col("procedure_code") == "UNKNOWN", "UNKNOWN").otherwise("real_code").alias("procedure_code_type")
).count().show()

# How many procedure_codes from claims actually exist in cost table?
real_proc_claims = claims.filter(F.col("procedure_code") != "UNKNOWN").select("procedure_code").distinct()
cost_codes       = cost.select("procedure_code").distinct()
matchable = real_proc_claims.join(cost_codes, on="procedure_code", how="inner").count()
print(f"procedure_codes in claims that match silver_cost: {matchable}")

# COMMAND ----------

# MAGIC %md
# MAGIC Compute billing_ratio
# MAGIC
# MAGIC Join silver_cost on proc_id = procedure_code.
# MAGIC billing_ratio is NULL when procecure_code = UNKNOWN 

# COMMAND ----------

# Left join on procedure_code (same column name in both tables now)
df = claims.join(cost, on="procedure_code", how="left")

# Compute billing_ratio safely (NULL when no cost benchmark available)
df = df.withColumn(
    "billing_ratio",
    F.when(
        F.col("expected_cost").isNotNull() & (F.col("expected_cost") > 0),
        F.round(F.col("billed_amount") / F.col("expected_cost"), 4)
    ).otherwise(F.lit(None).cast("double"))
)

print("billing_ratio computed. Sample:")
df.select("claim_id", "procedure_code", "billed_amount", "expected_cost", "billing_ratio") \
  .filter(F.col("billing_ratio").isNotNull()) \
  .show(10, truncate=False)

print(f"\nbilling_ratio NULL count (expected for UNKNOWN procedure_code): "
      f"{df.filter(F.col('billing_ratio').isNull()).count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute final_risk_score
# MAGIC
# MAGIC final_risk_score = pre_risk_score  +  overbilling points
# MAGIC
# MAGIC pre_risk_score was already computed in the Silver notebook and covers:
# MAGIC ghost claims, missing proc, missing billed, missing diag, missing provider, missing date.
# MAGIC
# MAGIC Gold adds the **billing ratio penalty** on top:
# MAGIC
# MAGIC | Condition | Points |
# MAGIC |---|:---:|
# MAGIC | billing_ratio > 2.0× | +4 |
# MAGIC | billing_ratio 1.5–2.0× | +2 |
# MAGIC | billing_ratio not available (UNKNOWN proc) | +0 |

# COMMAND ----------

df = df.withColumn(
    "billing_points",
    F.when(
        F.col("billing_ratio").isNotNull() & (F.col("billing_ratio") > 2.0),
        F.lit(4)
    ).when(
        F.col("billing_ratio").isNotNull() &
        (F.col("billing_ratio") > 1.5) & (F.col("billing_ratio") <= 2.0),
        F.lit(2)
    ).otherwise(F.lit(0))
    .cast("int")
)

df = df.withColumn(
    "final_risk_score",
    (F.col("pre_risk_score") + F.col("billing_points")).cast("int")
)

# Assign final risk tier label based on final score
df = df.withColumn(
    "risk_tier",
    F.when(F.col("final_risk_score") == 0,   "None")
     .when(F.col("final_risk_score") <= 2,   "Low")
     .when(F.col("final_risk_score") <= 4,   "Medium")
     .when(F.col("final_risk_score") <= 6,   "High")
     .otherwise("Critical")
)

print("final_risk_score computed. Distribution:")
df.groupBy("final_risk_score", "risk_tier") \
  .count() \
  .withColumnRenamed("count", "num_claims") \
  .orderBy("final_risk_score") \
  .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Threshold + 3% Noise
# MAGIC
# MAGIC Hard rule: final_risk_score >= 5 → **Denied (0)**, else **Approved (1)**.
# MAGIC
# MAGIC Then 5% of labels are randomly flipped to simulate rare real-world errors.
# MAGIC Theoretical accuracy ceiling: **~99.5%**.

# COMMAND ----------

# Hard threshold
df = df.withColumn(
    "_base_label",
    F.when(F.col("final_risk_score") >= DENY_THRESHOLD, 0)   # 0 = Denied
     .otherwise(1)                                             # 1 = Approved
    .cast("int")
)

# 0.5% symmetric noise
df = df.withColumn("_rand_noise", F.rand(seed=RANDOM_SEED)) \
       .withColumn(
           "denial_flag",
           F.when(F.col("_rand_noise") < NOISE_RATE, (1 - F.col("_base_label")).cast("int"))
            .otherwise(F.col("_base_label").cast("int"))
       )

total    = df.count()
denied   = df.filter(F.col("denial_flag") == 0).count()
approved = df.filter(F.col("denial_flag") == 1).count()

print(f"Labels assigned.")
print(f"  Total    : {total:,}")
print(f"  Denied(0): {denied:,}  ({100*denied/total:.1f}%)")
print(f"  Approved : {approved:,}  ({100*approved/total:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Final Gold Table
# MAGIC  **Only `claim_id` and `denial_flag` are written.**
# MAGIC All intermediate computation columns (`pre_risk_score`, `final_risk_score`,
# MAGIC `billing_ratio`, `billing_points`, `risk_tier`) are dropped here.
# MAGIC They were used to *construct* the label and must not be available to ML models
# MAGIC as features — that would be target leakage.

# COMMAND ----------

# Keep only the identifier + the label
leakage_cols = [
    "_base_label", "_rand_noise",
    "billing_points", "expected_cost",
    "billing_ratio",
    "pre_risk_score", "final_risk_score",
    "risk_tier",
]
gold_df = df.select("claim_id","patient_id","provider_id" ,"diagnosis_code","procedure_code","billed_amount","is_billed_missing","date","denial_flag")

print(f"Gold table columns ({len(gold_df.columns)}): {gold_df.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Gold Delta Table

# COMMAND ----------

gold_df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.gold.gold_claims_labeled")

print(" workspace.gold.gold_claims_labeled written.")

# Verify
spark.table("workspace.default.gold_claims_labeled") \
     .select("claim_id","patient_id","provider_id" ,"diagnosis_code","procedure_code","billed_amount","is_billed_missing","date","denial_flag") \
     .show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Report

# COMMAND ----------

result = spark.table("workspace.gold.gold_claims_labeled")
total    = result.count()
denied   = result.filter(F.col("denial_flag") == 0).count()
approved = result.filter(F.col("denial_flag") == 1).count()

print("=" * 55)
print("  STEP 9: LABEL GENERATION REPORT")
print("=" * 55)
print(f"  Table       : workspace.default.gold_claims_labeled")
print(f"  Columns     : claim_id, denial_flag")
print(f"  Threshold   : final_risk_score >= {DENY_THRESHOLD} → Denied (used internally, not stored)")
print(f"  Noise rate  : {NOISE_RATE*100}%  (accuracy ceiling ~{(1-NOISE_RATE)*100:.1f}%)")
print(f"  Noise rate  : {NOISE_RATE*100}%  (accuracy ceiling ~{(1-NOISE_RATE)*100:.1f}%)")
print(f"  Seed        : {RANDOM_SEED}")
print()
print(f"  Total rows  : {total:,}")
print(f"  Denied  (0) : {denied:,}  ({100*denied/total:.1f}%)")
print(f"  Approved(1) : {approved:,}  ({100*approved/total:.1f}%)") 
print()
print(" NOTE: billing_ratio / final_risk_score / risk_tier are NOT stored in the output table.")
print("They were used only to construct the label and are excluded to prevent ML leakage.")
print("=" * 55)
