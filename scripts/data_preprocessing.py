from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
from pyspark.ml.feature import StringIndexer

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Retail Data Processing") \
    .getOrCreate()

# Load data from Google Cloud Storage
transactions_df = spark.read.csv("gs://retail-spending-prediction/transactions.csv", header=True, inferSchema=True)
customers_df = spark.read.csv("gs://retail-spending-prediction/customers.csv", header=True, inferSchema=True)

# Data Wrangling: Join transaction data with customer demographic data
merged_df = transactions_df.join(customers_df, transactions_df.CustomerID == customers_df.CustomerID, "inner")

# Data Transformation: Create a new column to classify high vs. low spender based on amount spent
average_spend = merged_df.select(avg("Amount")).first()[0]
merged_df = merged_df.withColumn("SpenderClass", (col("Amount") > average_spend).cast("int"))

# Encode 'Gender' column as numeric
indexer = StringIndexer(inputCol="Gender", outputCol="GenderIndex")
indexed_df = indexer.fit(merged_df).transform(merged_df)

# Select features for machine learning
features_df = indexed_df.select("Age", "Amount", "SpenderClass", "GenderIndex", "AgeGroup")

# Save cleaned data back to Google Cloud Storage
features_df.write.csv("gs://retail-spending-prediction/cleaned_data.csv", header=True)

# Stop the Spark session
spark.stop()
