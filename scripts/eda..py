from google.cloud import bigquery
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Authenticate and connect to Google BigQuery
client = bigquery.Client()

# Querying store transactions data from BigQuery
transactions_query = """
SELECT CustomerID, ProductID, Amount, Date
FROM `your_project_id.your_dataset_id.transactions`
"""
transactions_df = client.query(transactions_query).to_dataframe()

# Querying customer demographic data from BigQuery
customers_query = """
SELECT CustomerID, Age, Salary, Gender, Country
FROM `your_project_id.your_dataset_id.customers`
"""
customers_df = client.query(customers_query).to_dataframe()

# Data Preprocessing: Merging datasets on CustomerID
merged_df = pd.merge(transactions_df, customers_df, on='CustomerID', how='inner')

# Handling missing values: Drop rows with NaN values
merged_df = merged_df.dropna()

# Feature Engineering: Create 'High Spender' vs 'Low Spender' column
merged_df['High_Spender'] = (merged_df['Amount'] > merged_df['Amount'].median()).astype(int)

# Key visualizations from the data:

# 1. Revenue by Country
revenue_by_country = merged_df.groupby('Country')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Country', y='Amount', data=revenue_by_country, palette='Blues')
plt.title('Revenue by Country', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Total Revenue (Amount)', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# 2. Spending by Gender
spending_by_gender = merged_df.groupby('Gender')['Amount'].sum().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(x='Gender', y='Amount', data=spending_by_gender, palette='Set1')
plt.title('Spending by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Total Spending (Amount)', fontsize=12)
plt.show()

# 3. Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(merged_df['Age'], bins=20, kde=True, color='purple')
plt.title('Age Distribution', fontsize=16)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# 4. Salary vs Spending
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Salary', y='Amount', data=merged_df, hue='Gender', palette='Set2')
plt.title('Salary vs Spending', fontsize=16)
plt.xlabel('Salary', fontsize=12)
plt.ylabel('Spending (Amount)', fontsize=12)
plt.show()

# 5. Top 10 Customers by Spending
top_customers_query = """
SELECT CustomerID, SUM(Amount) AS total_spending
FROM `your_project_id.your_dataset_id.transactions`
GROUP BY CustomerID
ORDER BY total_spending DESC
LIMIT 10
"""
top_customers = client.query(top_customers_query).to_dataframe()

plt.figure(figsize=(10, 6))
sns.barplot(x='CustomerID', y='total_spending', data=top_customers, palette='viridis')
plt.title('Top 10 Customers by Spending', fontsize=16)
plt.xlabel('Customer ID', fontsize=12)
plt.ylabel('Total Spending (Amount)', fontsize=12)
plt.xticks(rotation=90)
plt.show()

# 6. Top 10 Products by Revenue
top_products_query = """
SELECT ProductID, SUM(Amount) AS total_revenue
FROM `your_project_id.your_dataset_id.transactions`
GROUP BY ProductID
ORDER BY total_revenue DESC
LIMIT 10
"""
top_products = client.query(top_products_query).to_dataframe()

plt.figure(figsize=(10, 6))
sns.barplot(x='ProductID', y='total_revenue', data=top_products, palette='inferno')
plt.title('Top 10 Products by Revenue', fontsize=16)
plt.xlabel('Product ID', fontsize=12)
plt.ylabel('Revenue (Amount)', fontsize=12)
plt.xticks(rotation=90)
plt.show()

# 7. Monthly Sales Trend: Sales over the months
monthly_sales_query = """
SELECT EXTRACT(MONTH FROM Date) AS Month, SUM(Amount) AS total_sales
FROM `your_project_id.your_dataset_id.transactions`
GROUP BY Month
ORDER BY Month
"""
monthly_sales = client.query(monthly_sales_query).to_dataframe()

plt.figure(figsize=(10, 6))
sns.lineplot(x='Month', y='total_sales', data=monthly_sales, marker='o', color='teal')
plt.title('Monthly Sales Trend', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Sales (Amount)', fontsize=12)
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()
