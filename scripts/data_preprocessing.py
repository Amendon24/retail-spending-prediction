from google.cloud import bigquery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Authenticate and connect to Google BigQuery
client = bigquery.Client()

# Querying store transactions data from BigQuery
transactions_query = """
SELECT CustomerID, ProductID, Amount, Date
FROM `your_project_id.your_dataset_id.transactions`
"""
# Fetching the data into a DataFrame
transactions_df = client.query(transactions_query).to_dataframe()

# Querying customer demographic data from BigQuery
customers_query = """
SELECT CustomerID, Age, Salary, Gender, Country
FROM `your_project_id.your_dataset_id.customers`
"""
# Fetching customer data into a DataFrame
customers_df = client.query(customers_query).to_dataframe()

# Data Preprocessing: Merging the two datasets on CustomerID
merged_df = pd.merge(transactions_df, customers_df, on='CustomerID', how='inner')

# Handle missing values: Drop rows with NaN values in the merged dataset
merged_df = merged_df.dropna()

# Feature Engineering: Create a column to classify high vs low spenders
merged_df['High_Spender'] = (merged_df['Amount'] > merged_df['Amount'].median()).astype(int)

# Save the preprocessed data to Google Cloud Storage
merged_df.to_csv('gs://your_bucket_id/cleaned_data.csv', index=False)

# Displaying some rows to confirm preprocessing
print(merged_df.head())
