# Databricks notebook source
# MAGIC %md
# MAGIC # Data Cleaning & Trusted Dataset
# MAGIC
# MAGIC ## Chained Data Loss — Dependency Map
# MAGIC
# MAGIC Understanding this is critical before any imputation decision:
# MAGIC
# MAGIC claims.procedure_code  ──► cost.procedure_code ──► expected_cost, region
# MAGIC
# MAGIC claims.diag_code ──► diagnosis.diagnosis_code ──► category, severity
# MAGIC
# MAGIC claims.provider_id ──► providers.provider_id ──► location (≈ region)
# MAGIC
# MAGIC **Note:** providers.location and cost.region store the same Indian cities.
# MAGIC So when procedure_code is null and we cannot join to cost, we recover region
# MAGIC by joining claims.provider_id → providers.location.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Setup

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, DateType
from datetime import datetime

SILVER_VERSION = "1.0"
CLEANED_AT     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

try:
    # Load Bronze tables
    df_claims    = spark.table("default.bronze_claims_raw")
    df_providers = spark.table("default.bronze_provider_raw")
    df_diagnosis = spark.table("default.bronze_diagnosis_raw")
    df_cost      = spark.table("default.bronze_cost_raw")
except Exception as e:
    print(f"Error loading Bronze tables: {e}")
    raise

# Drop Bronze-only metadata columns
BRONZE_META = ["ingestion_time", "source_file"]
def drop_bronze_meta(df):
    cols_to_drop = [c for c in BRONZE_META if c in df.columns]
    return df.drop(*cols_to_drop)

try:
    df_claims    = drop_bronze_meta(df_claims)
    df_providers = drop_bronze_meta(df_providers)
    df_diagnosis = drop_bronze_meta(df_diagnosis)
    df_cost      = drop_bronze_meta(df_cost)
except Exception as e:
    print(f"Error dropping Bronze metadata columns: {e}")
    raise

print(f"Bronze tables loaded. Cleaned at: {CLEANED_AT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Clean Providers Table

# COMMAND ----------

# Dedup
before_dedup = df_providers.count()
df_providers = df_providers.dropDuplicates(["provider_id"])
print(f"Provider duplicates removed: {before_dedup - df_providers.count()}")

# Standardise
df_providers = df_providers \
    .withColumn("provider_id", F.upper(F.trim(F.col("provider_id")))) \
    .withColumn("doctor_name", F.initcap(F.trim(F.col("doctor_name")))) \
    .withColumn("specialty",   F.initcap(F.trim(F.col("specialty")))) \ 
    .withColumn("location",    F.initcap(F.trim(F.col("location"))))

# Handle missing location — cannot statistically impute a city
missing_loc = df_providers.filter(F.col("location").isNull()).count()
df_providers = df_providers \
    .withColumn("location", F.coalesce(F.col("location"), F.lit("Unknown")))
print(f"Missing location filled with 'Unknown': {missing_loc}")

# Handle missing specialty
missing_spec = df_providers.filter(F.col("specialty").isNull()).count()
df_providers = df_providers \
    .withColumn("specialty", F.coalesce(F.col("specialty"), F.lit("Unknown")))
print(f"Missing specialty filled with 'Unknown': {missing_spec}")

df_silver_providers = df_providers \
    .withColumn("cleaned_at",     F.lit(CLEANED_AT)) \
    .withColumn("silver_version", F.lit(SILVER_VERSION))

print(f"Silver Providers ready: {df_silver_providers.count()} rows")
display(df_silver_providers.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Clean Diagnosis Table

# COMMAND ----------

# Dedup
before_dedup = df_diagnosis.count()
df_diagnosis = df_diagnosis.dropDuplicates(["diagnosis_code"])
print(f"Diagnosis duplicates removed: {before_dedup - df_diagnosis.count()}")

# Standardise
df_diagnosis = df_diagnosis \
    .withColumn("diagnosis_code", F.upper(F.trim(F.col("diagnosis_code")))) \
    .withColumn("category",       F.initcap(F.trim(F.col("category")))) \
    .withColumn(
        "severity",
        # Normalise all variants into 4 standard levels
        F.when(F.upper(F.trim(F.col("severity"))).isin("MILD", "LOW"),                "Mild")
         .when(F.upper(F.trim(F.col("severity"))).isin("MODERATE", "MEDIUM"),         "Moderate")
         .when(F.upper(F.trim(F.col("severity"))).isin("SEVERE", "HIGH", "CRITICAL"), "Severe")
         .otherwise("Unknown")
    )

# Handle missing category
missing_cat = df_diagnosis.filter(F.col("category").isNull()).count()
df_diagnosis = df_diagnosis \
    .withColumn("category", F.coalesce(F.col("category"), F.lit("Unknown")))
print(f"Missing category filled: {missing_cat}")

df_silver_diagnosis = df_diagnosis \
    .withColumn("cleaned_at",     F.lit(CLEANED_AT)) \
    .withColumn("silver_version", F.lit(SILVER_VERSION))

print(f"Silver Diagnosis ready: {df_silver_diagnosis.count()} rows")
display(df_silver_diagnosis.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Clean Cost Table

# COMMAND ----------

# Dedup
before_dedup = df_cost.count()
df_cost = df_cost.dropDuplicates(["procedure_code"])
print(f"Cost duplicates removed: {before_dedup - df_cost.count()}")

# Type casting
df_cost = df_cost \
    .withColumn("expected_cost", F.col("expected_cost").cast(DoubleType())) \
    .withColumn("average_cost",  F.col("average_cost").cast(DoubleType()))

# Standardise
df_cost = df_cost \
    .withColumn("procedure_code", F.upper(F.trim(F.col("procedure_code")))) \
    .withColumn("region",         F.initcap(F.trim(F.col("region"))))

# Impute costs using per-region median (costs vary by geography)
missing_exp = df_cost.filter(F.col("expected_cost").isNull()).count()
missing_avg = df_cost.filter(F.col("average_cost").isNull()).count()

window_region = Window.partitionBy("region")
df_cost = df_cost \
    .withColumn("region_med_exp", F.percentile_approx("expected_cost", 0.5).over(window_region)) \
    .withColumn("region_med_avg", F.percentile_approx("average_cost",  0.5).over(window_region))

global_exp = df_cost.agg(F.percentile_approx("expected_cost", 0.5).alias("m")).collect()[0]["m"]
global_avg = df_cost.agg(F.percentile_approx("average_cost",  0.5).alias("m")).collect()[0]["m"]

df_cost = df_cost \
    .withColumn("expected_cost",
        F.when(F.col("expected_cost").isNull(),
               F.coalesce(F.col("region_med_exp"), F.lit(global_exp)))
        .otherwise(F.col("expected_cost"))) \
    .withColumn("average_cost",
        F.when(F.col("average_cost").isNull(),
               F.coalesce(F.col("region_med_avg"), F.lit(global_avg)))
        .otherwise(F.col("average_cost"))) \
    .withColumn("expected_cost", F.round("expected_cost", 2)) \
    .withColumn("average_cost",  F.round("average_cost", 2)) \
    .drop("region_med_exp", "region_med_avg")

missing_region = df_cost.filter(F.col("region").isNull()).count()
df_cost = df_cost.withColumn("region", F.coalesce(F.col("region"), F.lit("Unknown")))

print(f"Missing expected_cost filled (per-region median): {missing_exp}")
print(f"Missing average_cost  filled (per-region median): {missing_avg}")
print(f"Missing region        filled: {missing_region}")

df_silver_cost = df_cost \
    .withColumn("cleaned_at",     F.lit(CLEANED_AT)) \
    .withColumn("silver_version", F.lit(SILVER_VERSION))

print(f"Silver Cost ready: {df_silver_cost.count()} rows")
display(df_silver_cost.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Clean Claims Table
# MAGIC
# MAGIC ### Chained Data Loss Strategy
# MAGIC
# MAGIC **Case A — procedure_code present:**
# MAGIC Join to cost table → get expected_cost and region directly.
# MAGIC Impute billed_amount using per-procedure-code median.
# MAGIC
# MAGIC **Case B — procedure_code missing, provider_id present:**
# MAGIC Cannot get expected_cost (no procedure benchmark — never guess this).
# MAGIC Recover region from providers.location (same geography).
# MAGIC Impute billed_amount using overall median (no proc group available).
# MAGIC Flag: is_proc_missing = True.
# MAGIC
# MAGIC **Case C — Both billed_amount AND proc_id missing:**
# MAGIC Impute billed_amount with overall median.
# MAGIC Leave expected_cost = NULL (cannot guess a benchmark).
# MAGIC Flag: is_both_missing = True → triggers highest risk score.

# COMMAND ----------

# Step 1: Dedup on claim_id
before_dedup = df_claims.count()
df_claims = df_claims.dropDuplicates(["claim_id"])
print(f"Claims duplicates removed: {before_dedup - df_claims.count()}")

# Step 2: Type casting
df_claims = df_claims \
    .withColumn("billed_amount", F.col("billed_amount").cast(DoubleType())) \
    .withColumn("date",          F.to_date(F.col("date"), "yyyy-MM-dd"))

# Step 3: Standardise codes
for code_col in ["claim_id", "patient_id", "provider_id", "diagnosis_code", "procedure_code"]:
    if code_col in df_claims.columns:
        df_claims = df_claims.withColumn(code_col, F.upper(F.trim(F.col(code_col))))

# Step 4: Replace negative billed amounts with NULL (data entry error)
neg_count = df_claims.filter(F.col("billed_amount") < 0).count()
df_claims = df_claims.withColumn(
    "billed_amount",
    F.when(F.col("billed_amount") < 0, None).otherwise(F.col("billed_amount"))
)
print(f"Negative billed_amount rows nulled: {neg_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Flag all missing columns BEFORE any imputation

# COMMAND ----------

# Create transparency flags first — always reflect the RAW state from Bronze
df_claims = df_claims \
    .withColumn("is_billed_missing",   F.col("billed_amount").isNull()) \
    .withColumn("is_proc_missing",     F.col("procedure_code").isNull()) \
    .withColumn("is_diag_missing",     F.col("diagnosis_code").isNull()) \
    .withColumn("is_provider_missing", F.col("provider_id").isNull()) \
    .withColumn("is_date_missing",     F.col("date").isNull()) \
    .withColumn("is_both_missing",     F.col("billed_amount").isNull() & F.col("procedure_code").isNull())

# Print counts before any changes
print("=== Missing value counts (RAW from Bronze) ===")
for flag in ["is_billed_missing", "is_proc_missing", "is_diag_missing",
             "is_provider_missing", "is_date_missing", "is_both_missing"]:
    n = df_claims.filter(F.col(flag)).count()
    print(f"  {flag}: {n}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Impute billed_amount
# MAGIC Strategy: Grouped median by procedure_code → fall back to overall median when procedure_code is null

# COMMAND ----------

# Compute per-procedure median from non-null rows
window_proc = Window.partitionBy("procedure_code")
df_claims = df_claims.withColumn(
    "proc_median_billed",
    F.percentile_approx("billed_amount", 0.5).over(window_proc)
)

# Compute overall median as fallback (for rows where procedure_code is also null)
global_median_billed = df_claims.agg(
    F.percentile_approx("billed_amount", 0.5).alias("m")
).collect()[0]["m"]
print(f"Overall median billed_amount (fallback): {global_median_billed}")

# Apply: proc-group median → fallback to global median → round to 2dp
df_claims = df_claims \
    .withColumn("billed_amount",
        F.when(F.col("billed_amount").isNull(),
               F.coalesce(F.col("proc_median_billed"), F.lit(global_median_billed)))
        .otherwise(F.col("billed_amount"))) \
    .withColumn("billed_amount", F.round("billed_amount", 2)) \
    .drop("proc_median_billed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Handle missing categorical codes (diagnosis, procedure, provider)
# MAGIC ICD-10 and CPT codes CANNOT be statistically imputed — sentinel UNKNOWN is used.

# COMMAND ----------

# Fill categorical codes with UNKNOWN sentinel
df_claims = df_claims \
    .withColumn("diagnosis_code",  F.coalesce(F.col("diagnosis_code"),  F.lit("UNKNOWN"))) \
    .withColumn("procedure_code",  F.coalesce(F.col("procedure_code"),  F.lit("UNKNOWN"))) \
    .withColumn("provider_id",     F.coalesce(F.col("provider_id"),     F.lit("UNKNOWN")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Region Recovery (3-Level Fallback)
# MAGIC
# MAGIC **Level 1 (Best):** `procedure_code` exists → join to cost table → get `region` directly.
# MAGIC
# MAGIC **Level 2 (Recovery):** `procedure_code` missing but `provider_id` present → join to providers → use `location` as region proxy.
# MAGIC
# MAGIC **Level 3 (Total Blackout):** Both `procedure_code` AND `provider_id` missing → set region to 'Unknown'. Cannot recover. Highest risk.

# COMMAND ----------

# Build a slim provider lookup: provider_id → location (already cleaned)
provider_lookup = df_silver_providers.select(
    F.col("provider_id"),
    F.col("location").alias("provider_location")
)

# Left-join to get provider location for every claim
# NOTE: When provider_id was NULL in Bronze, it was filled with 'UNKNOWN' above.
# 'UNKNOWN' will not match any real provider_id, so provider_location will be NULL for those rows.
df_claims = df_claims.join(provider_lookup, on="provider_id", how="left")

# Flag the "Total Blackout" case: both proc_id AND provider_id were originally missing
df_claims = df_claims.withColumn(
    "is_total_blackout",
    F.col("is_proc_missing") & F.col("is_provider_missing")
)

total_blackout_count = df_claims.filter(F.col("is_total_blackout")).count() 
print(f"Total Blackout rows (proc AND provider both missing): {total_blackout_count}")

# 3-Level COALESCE for final_region:
#   Level 1: region from cost table join (done at Gold layer — not available here yet)
#   Level 2: location from provider table (recovered when procecdure_code is missing)
#   Level 3: 'Unknown' literal (total blackout — neither proc nor provider available)
# At Silver layer we store the recovery attempt; Gold will apply Level 1.
df_claims = df_claims.withColumn(
    "recovered_region",
    F.when(
        # If procedure_code was missing, attempt Level 2 recovery from provider location
        F.col("is_proc_missing"),
        F.coalesce(
            F.col("provider_location"),  # Level 2: from provider
            F.lit("Unknown")             # Level 3: total blackout fallback
        )
    ).otherwise(None)  # procedure_code present → Gold layer will join cost table for real region
)

print("3-level region recovery complete. 'recovered_region' column added.")
print("  → None    : procedure_code present (Gold will get region from cost table)")
print("  → City    : procedure_code missing but provider known (recovered from provider.location)")
print("  → Unknown : both procedure_code AND provider_id missing (Total Blackout)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5e — Pre-compute Risk Score
# MAGIC This score is stored in Silver so Gold/ML can use it directly.
# MAGIC Higher score = higher probability of denial.

# COMMAND ----------

df_claims = df_claims.withColumn(
    "pre_risk_score",
    # ── Primary financial signal (mutually exclusive base scores) ──────────────
    # Worst case: both billed_amount AND procedure_code missing (+5)
    F.when(F.col("is_both_missing"), F.lit(5)) 
    # procedure_code missing only — billed_amount was present (+3)
    .when(F.col("is_proc_missing") & ~F.col("is_billed_missing"), F.lit(3))
    # billed_amount missing only — procedure_code was present (+2)
    .when(F.col("is_billed_missing") & ~F.col("is_proc_missing"), F.lit(2))
    .otherwise(F.lit(0))
    # ── Additive signals stacked on top ────────────────────────────────────────
    # Total blackout: both procedure_code AND provider_id missing (+1 extra penalty)
    + F.when(F.col("is_total_blackout"),   F.lit(1)).otherwise(F.lit(0))
    # Clinical justification missing
    + F.when(F.col("is_diag_missing"),     F.lit(2)).otherwise(F.lit(0))
    # Unknown submitter — identity risk
    + F.when(F.col("is_provider_missing"), F.lit(2)).otherwise(F.lit(0))
    # Administrative issue — missing date
    + F.when(F.col("is_date_missing"),     F.lit(1)).otherwise(F.lit(0))
    # NOTE: billing_ratio-based points (+4 if >2x, +2 if >1.5x) are added
    # in the Gold notebook after joining silver_claims with silver_cost.
)

print("Pre-risk score assigned.")

# COMMAND ----------

# Add Silver metadata
df_silver_claims = df_claims \
    .withColumn("cleaned_at",     F.lit(CLEANED_AT)) \
    .withColumn("silver_version", F.lit(SILVER_VERSION))

print(f"Silver Claims ready: {df_silver_claims.count()} rows, {len(df_silver_claims.columns)} cols")
display(df_silver_claims.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Write Silver Delta Tables

# COMMAND ----------

silver_tables = [
    (df_silver_claims,    "workspace.silver.silver_claims"),
    (df_silver_providers, "workspace.silver.silver_provider"),
    (df_silver_diagnosis, "workspace.silver.silver_diagnosis"),
    (df_silver_cost,      "workspace.silver.silver_cost"),
]

for df, table_name in silver_tables:
    df.write.format("delta").mode("overwrite").saveAsTable(table_name)
    print(f"Saved → {table_name}  ({df.count()} rows)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Null Checks After Cleaning

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   'claim_id'        AS col, COUNT(CASE WHEN claim_id IS NULL THEN 1 END)           AS nulls FROM workspace.silver.silver_claims UNION ALL
# MAGIC SELECT 'patient_id',        COUNT(CASE WHEN patient_id IS NULL THEN 1 END)         FROM workspace.silver.silver_claims UNION ALL
# MAGIC SELECT 'provider_id',       COUNT(CASE WHEN provider_id IS NULL THEN 1 END)        FROM workspace.silver.silver_claims UNION ALL
# MAGIC SELECT 'diagnosis_code',    COUNT(CASE WHEN diagnosis_code IS NULL THEN 1 END)     FROM workspace.silver.silver_claims UNION ALL
# MAGIC SELECT 'procedure_code',    COUNT(CASE WHEN procedure_code IS NULL THEN 1 END)     FROM workspace.silver.silver_claims UNION ALL
# MAGIC SELECT 'billed_amount',     COUNT(CASE WHEN billed_amount IS NULL THEN 1 END)      FROM workspace.silver.silver_claims UNION ALL
# MAGIC SELECT 'date',              COUNT(CASE WHEN date IS NULL THEN 1 END)               FROM workspace.silver.silver_claims
# MAGIC ORDER BY nulls DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Primary Key Uniqueness

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'silver_claims'    AS tbl, COUNT(*) - COUNT(DISTINCT claim_id)        AS dupe_pks FROM workspace.silver.silver_claims    UNION ALL
# MAGIC SELECT 'silver_provider',          COUNT(*) - COUNT(DISTINCT provider_id)    AS dupe_pks FROM workspace.silver.silver_provider  UNION ALL
# MAGIC SELECT 'silver_diagnosis',         COUNT(*) - COUNT(DISTINCT diagnosis_code) AS dupe_pks FROM workspace.silver.silver_diagnosis UNION ALL
# MAGIC SELECT 'silver_cost',              COUNT(*) - COUNT(DISTINCT procedure_code) AS dupe_pks FROM workspace.silver.silver_cost;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre-Risk Score Distribution

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT pre_risk_score, COUNT(*) AS claim_count,
# MAGIC        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
# MAGIC FROM workspace.silver.silver_claims
# MAGIC GROUP BY pre_risk_score
# MAGIC ORDER BY pre_risk_score ASC;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imputation Transparency Summary

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   COUNT(*)                                                                    AS total_claims,
# MAGIC   SUM(CASE WHEN is_billed_missing   = true THEN 1 END)                      AS billed_imputed,
# MAGIC   SUM(CASE WHEN is_proc_missing     = true THEN 1 END)                      AS proc_unknown,
# MAGIC   SUM(CASE WHEN is_diag_missing     = true THEN 1 END)                      AS diag_unknown,
# MAGIC   SUM(CASE WHEN is_provider_missing = true THEN 1 END)                      AS provider_unknown,
# MAGIC   SUM(CASE WHEN is_both_missing     = true THEN 1 END)                      AS both_bill_and_proc_missing,
# MAGIC   SUM(CASE WHEN recovered_region IS NOT NULL THEN 1 END)                    AS region_recovered_from_provider
# MAGIC FROM workspace.silver.silver_claims;
