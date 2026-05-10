# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering 

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.window import Window

print("Imports loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Silver Tables

# COMMAND ----------

try:
    provider  = spark.table("workspace.silver.silver_provider")
    diagnosis = spark.table("workspace.silver.silver_diagnosis")
    gold      = spark.table("workspace.gold.gold_claims_labeled") \
                     .select("claim_id", "denial_flag")  # only the label — no leakage columns
    cost = spark.table("workspace.silver.silver_cost")

    print(f"silver_provider  : {provider.count():,} rows")
    print(f"silver_diagnosis : {diagnosis.count():,} rows")
    print(f"silver_cost      : {cost.count():,} rows")
    print(f"gold_labeled     : {gold.count():,} rows")
except Exception as e:
    print(f"Error loading tables: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build gold_claim_base Table

# COMMAND ----------


spark.sql("""

    SELECT

        provider_id,

        specialty   AS provider_specialty

    FROM workspace.silver.silver_provider

""").createOrReplaceTempView("provider_sel")

spark.sql("""

    SELECT

        diagnosis_code,

        category    AS diag_category,

        severity    AS diag_severity

    FROM workspace.silver.silver_diagnosis

""").createOrReplaceTempView("diagnosis_sel")

spark.sql("""

    SELECT

        procedure_code,

        average_cost,

        expected_cost,

        region      AS proc_region

    FROM workspace.silver.silver_cost

""").createOrReplaceTempView("cost_sel")

# Build gold_claim_base 

base_sql = """

SELECT

    g.*,

    p.provider_specialty,

    d.diag_category,

    d.diag_severity,

    co.average_cost,

    co.expected_cost,

    co.proc_region

FROM workspace.gold.gold_claims_labeled  g

LEFT JOIN provider_sel  p   ON g.provider_id     = p.provider_id

LEFT JOIN diagnosis_sel d   ON g.diagnosis_code  = d.diagnosis_code

LEFT JOIN cost_sel      co  ON g.procedure_code  = co.procedure_code

"""

base = spark.sql(base_sql)

print(f"gold_claim_base  : {base.count():,} rows | {len(base.columns)} columns")

display(base.limit(5))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Write gold_claim_base

# COMMAND ----------

# DBTITLE 1,Cell 9
base.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.gold.gold_claim_base")

print(" workspace.gold.gold_claim_base written.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering
# MAGIC
# MAGIC ### Feature Groups
# MAGIC
# MAGIC | Group | Features | Domain Rationale |
# MAGIC |---|---|---|
# MAGIC | **Cost** | billing_ratio, `cost_diff, high_cost_flag | Overbilling is #1 denial cause |
# MAGIC | **Provider** | provider_claim_count, provider_denial_rate, provider_risk_score | Fraudulent providers show volume patterns |
# MAGIC | **Diagnosis** | severity_score, diag_claim_count | High-severity + rare combos flagged |
# MAGIC | **Claim integrity** | pre_risk_score, final_risk_score, is_proc_missing, is_diag_missing, is_billed_missing | Data quality signals |
# MAGIC | **Categorical (encoded)** | provider_specialty_enc, diag_category_enc, risk_tier_enc | Label-encoded for tree models |
# MAGIC | **Temporal** | claim_age_days | Older claims may be stale/fraudulent |

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import DoubleType, IntegerType

df = base

# A. Cost features

# billing_ratio: billed_amount / expected_cost
# UNKNOWN procedure_code → expected_cost is null → fill with 1.0 (neutral signal)
df = df.withColumn(
    "billing_ratio",
    F.when(
        F.col("expected_cost").isNotNull() & (F.col("expected_cost") > 0),
        F.round(F.col("billed_amount") / F.col("expected_cost"), 4)
    ).otherwise(F.lit(1.0)).cast(DoubleType())
)

# cost_diff = billed_amount - expected_cost (positive → overbilled, negative → underbilled)
# Null expected_cost (UNKNOWN proc) → 0.0 (neutral)
df = df.withColumn(
    "cost_diff",
    F.when(
        F.col("expected_cost").isNotNull() & (F.col("expected_cost") > 0),
        F.col("billed_amount") - F.col("expected_cost")
    ).otherwise(F.lit(0.0)).cast(DoubleType())
)

# high_cost_flag: 1 if billing_ratio > 1.5
df = df.withColumn(
    "high_cost_flag",
    F.when(F.col("billing_ratio") > 1.5, 1).otherwise(0).cast(IntegerType())
)

# B. Provider features 

w_prov = Window.partitionBy("provider_id")

# provider_claim_count: total claims submitted by this provider
df = df.withColumn(
    "provider_claim_count",
    F.count("claim_id").over(w_prov).cast(IntegerType())
)

# provider_denial_rate: fraction of provider's claims where denial_flag == 0
# Schema: 0 = denied, 1 = approved → map 0 → 1.0 to compute denial rate
df = df.withColumn(
    "provider_denial_rate",
    F.round(
        F.avg(F.when(F.col("denial_flag") == 0, 1.0).otherwise(0.0)).over(w_prov),
        4
    ).cast(DoubleType())
)

# provider_avg_billed: average billed amount per provider (detects systematic overbilling)
df = df.withColumn(
    "provider_avg_billed",
    F.round(F.avg("billed_amount").over(w_prov), 4).cast(DoubleType())
)

# provider_risk_score: volume × denial_rate (high-volume + high-denial = highest risk)
df = df.withColumn(
    "provider_risk_score",
    F.round(
        F.col("provider_claim_count") * F.col("provider_denial_rate"), 4
    ).cast(DoubleType())
)

# C. Diagnosis features 

# severity_score: Mild=1, Severe=3, null/unknown=2 (neutral)
# Data contains only "Mild" and "Severe" — no "Low/Medium/High"
df = df.withColumn(
    "severity_score",
    F.when(F.upper(F.col("diag_severity")) == "MILD",   1)
     .when(F.upper(F.col("diag_severity")) == "SEVERE", 3)
     .otherwise(2)
     .cast(IntegerType())
)

# diag_claim_count: number of claims sharing this diagnosis code
# UNKNOWN diagnosis codes are all grouped together — expected behavior
w_diag = Window.partitionBy("diagnosis_code")
df = df.withColumn(
    "diag_claim_count",
    F.count("claim_id").over(w_diag).cast(IntegerType())
)

# D. Claim integrity flags 

# is_billed_missing: already present as boolean column in source data → cast to int
df = df.withColumn(
    "is_billed_missing",
    F.col("is_billed_missing").cast(IntegerType())
)

# is_proc_missing: 1 if procedure_code is null or "UNKNOWN"
# Derived from data — this column does NOT exist natively
df = df.withColumn(
    "is_proc_missing",
    F.when(
        F.col("procedure_code").isNull() | (F.upper(F.col("procedure_code")) == "UNKNOWN"), 1
    ).otherwise(0).cast(IntegerType())
)

# is_diag_missing: 1 if diagnosis_code is null or "UNKNOWN"
# Derived from data — this column does NOT exist natively
df = df.withColumn(
    "is_diag_missing",
    F.when(
        F.col("diagnosis_code").isNull() | (F.upper(F.col("diagnosis_code")) == "UNKNOWN"), 1
    ).otherwise(0).cast(IntegerType())
)

# missing_flag_count: total number of missing flags per claim (0–3)
# Useful aggregate integrity signal for the model
df = df.withColumn(
    "missing_flag_count",
    (F.col("is_billed_missing") + F.col("is_proc_missing") + F.col("is_diag_missing"))
    .cast(IntegerType())
)

# E. Temporal features 

max_date = df.agg(F.max("date")).collect()[0][0]

# claim_age_days: days between claim date and most recent claim in dataset
df = df.withColumn(
    "claim_age_days",
    F.datediff(F.lit(max_date), F.col("date")).cast(IntegerType())
)

# claim_month: month of claim (1–3 in this dataset; captures seasonal patterns)
df = df.withColumn(
    "claim_month",
    F.month(F.col("date")).cast(IntegerType())
)

# F. Regional features 

# proc_region_enc: label encode proc_region
# Regions present: Delhi, Mumbai, Bangalore, Hyderabad, Chennai, Ahmedabad, null
regions = [r.proc_region for r in
           df.select("proc_region").distinct().collect()
           if r.proc_region is not None]
region_map = {r: i for i, r in enumerate(sorted(regions))}
region_expr = F.create_map(
    *[x for pair in [(F.lit(k), F.lit(v)) for k, v in region_map.items()] for x in pair]
)
df = df.withColumn(
    "proc_region_enc",
    F.coalesce(region_expr[F.col("proc_region")], F.lit(-1)).cast(IntegerType())
    # -1 for UNKNOWN procedure rows where proc_region is null
)

# G. Categorical encoding

# provider_specialty_enc: label encode provider_specialty
# Values: Cardiology, General, Neurology, Orthopedic
specialties = [r.provider_specialty for r in
               df.select("provider_specialty").distinct().collect()
               if r.provider_specialty is not None]
specialty_map = {s: i for i, s in enumerate(sorted(specialties))}
specialty_expr = F.create_map(
    *[x for pair in [(F.lit(k), F.lit(v)) for k, v in specialty_map.items()] for x in pair]
)
df = df.withColumn(
    "provider_specialty_enc",
    F.coalesce(specialty_expr[F.col("provider_specialty")], F.lit(0)).cast(IntegerType())
)

# diag_category_enc: label encode diag_category
# Values: Bone, Diabetes, Fever, Heart, Skin, null
categories = [r.diag_category for r in
              df.select("diag_category").distinct().collect()
              if r.diag_category is not None]
category_map = {c: i for i, c in enumerate(sorted(categories))}
category_expr = F.create_map(
    *[x for pair in [(F.lit(k), F.lit(v)) for k, v in category_map.items()] for x in pair]
)
df = df.withColumn(
    "diag_category_enc",
    F.coalesce(category_expr[F.col("diag_category")], F.lit(0)).cast(IntegerType())
)

print("Feature engineering complete.")
print(f"Total rows: {df.count()} | Total columns: {len(df.columns)}")

df.select(
    "claim_id",
    # cost
    "billing_ratio", "cost_diff", "high_cost_flag",
    # provider
    "provider_claim_count", "provider_denial_rate",
    "provider_avg_billed", "provider_risk_score",
    # diagnosis
    "severity_score", "diag_claim_count",
    # integrity
    "is_billed_missing", "is_proc_missing", "is_diag_missing", "missing_flag_count",
    # temporal
    "claim_age_days", "claim_month",
    # categorical
    "proc_region_enc", "provider_specialty_enc", "diag_category_enc",
    # target
    "denial_flag"
).show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Select Final Feature Set & Write gold_claim_features

# COMMAND ----------

from pyspark.sql import functions as F

FEATURE_COLS = [
    # Cost group
    "billing_ratio",          # billed_amount / expected_cost
    "cost_diff",              # billed_amount - expected_cost
    "high_cost_flag",         # 1 if billing_ratio > 1.5
    # Provider group
    "provider_claim_count",   # total claims this provider submitted
    "provider_specialty_enc", # label-encoded specialty
    # Diagnosis group
    "severity_score",         # Mild=1 / Medium=2 / Severe=3
    "diag_claim_count",       # frequency of this diagnosis code
    "diag_category_enc",      # label-encoded diagnosis category
    # Claim integrity group
    "is_billed_missing",      # 1 if billed_amount was null in bronze
    "is_proc_missing",        # 1 if procedure_code was null in bronze
    "is_diag_missing",        # 1 if diagnosis_code was null in bronze
    # Temporal
    "claim_age_days",         # days since claim date
]

TARGET_COL = "denial_flag"
ID_COL     = "claim_id"


gold_features = df.select(ID_COL, *FEATURE_COLS, TARGET_COL)


print("Null check on feature set:")
print("-" * 45)

all_clean = True
for col in FEATURE_COLS + [TARGET_COL]:
    n = gold_features.filter(F.col(col).isNull()).count()
    status = "✅" if n == 0 else f"❌  {n} nulls"
    if n > 0:
        all_clean = False
    print(f"  {col:30s} {status}")  # BUG FIX: print was missing from original

print("-" * 45)

if not all_clean:
    raise ValueError("  Null values found in feature set — fix before writing.")  # BUG FIX: was outside the loop at wrong indent level

# ── Write to Delta table ───────────────────────────────────────────────────────

gold_features.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("workspace.gold.gold_claim_features")

print(f"\n workspace.gold.gold_claim_features written ({gold_features.count():,} rows, {len(FEATURE_COLS)} features).")
print(f"    Features : {FEATURE_COLS}")
print(f"    Target   : {TARGET_COL}")
