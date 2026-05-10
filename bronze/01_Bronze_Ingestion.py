# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer Ingestion (Databricks)
# MAGIC This notebook reads raw CSV data from the specified paths, adds metadata columns, and writes them as Managed Delta Tables for the bronze layer.

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, input_file_name

# 1. Define paths 
# Note: In Databricks Community Edition, Unity Catalog (Volumes) is not officially supported.
# If you uploaded via the DBFS UI, your paths might actually start with "dbfs:/FileStore/tables/".
# However, assuming you have configured these exact paths in your workspace:

raw_claims_path = "/Volumes/workspace/default/myvol/raw/claims/"
raw_providers_path = "/Volumes/workspace/default/myvol/raw/providers/"
raw_diagnosis_path = "/Volumes/workspace/default/myvol/raw/diagnosis/"
raw_cost_path = "/Volumes/workspace/default/myvol/raw/cost/"

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, col

def process_to_bronze(raw_path, table_name, file_format="csv"):
    print(f"Processing '{table_name}' from '{raw_path}'...")
    
    df = spark.read.format(file_format) \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(raw_path)
    
    df_with_meta = df \
        .withColumn("ingestion_time", current_timestamp()) \
        .withColumn("source_file", col("_metadata.file_path"))
    
    df_with_meta.write \
        .format("delta") \
        .mode("overwrite") \
        .option("mergeSchema", "true") \
        .saveAsTable(f"default.{table_name}")
        
    print(f"Successfully saved managed table: default.{table_name}")
    print(f"Row count: {df_with_meta.count()}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute Ingestion for All Datasets

# COMMAND ----------

# 3. Execute the pipeline
process_to_bronze(raw_claims_path, "bronze_claims_raw")
process_to_bronze(raw_providers_path, "bronze_provider_raw")
process_to_bronze(raw_diagnosis_path, "bronze_diagnosis_raw")
process_to_bronze(raw_cost_path, "bronze_cost_raw")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Tables
# MAGIC Let's query one of the tables to ensure data was loaded correctly.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM default.bronze_claims_raw LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM default.bronze_provider_raw LIMIT 5;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from default.bronze_cost_raw limit 5;
