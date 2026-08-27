import pandas as pd
import os

print("Pandas Pipeline Started... (Member 2)")

base_path = "data/reduced"
processed_path = "data/processed"
os.makedirs(processed_path, exist_ok=True)

# 1. Load
print("Loading data...")
articles = pd.read_csv(f"{base_path}/articles_reduced.csv")
customers = pd.read_csv(f"{base_path}/customers_reduced.csv")
transactions = pd.read_csv(f"{base_path}/transactions_reduced.csv")
print(f"Articles: {len(articles)}, Customers: {len(customers)}, Transactions: {len(transactions)}")

# 2. Cleaning - Member 2 Task
print("Cleaning...")
# Transactions
transactions = transactions.drop_duplicates()
transactions['t_dat'] = pd.to_datetime(transactions['t_dat'], errors='coerce')
transactions = transactions.dropna(subset=['t_dat'])
transactions = transactions[transactions['price'] > 0]

# Customers
customers = customers.drop_duplicates(subset=['customer_id'])
customers['age'] = customers['age'].fillna(30)
customers['club_member_status'] = customers['club_member_status'].astype(str).str.strip()

# Articles
articles = articles.drop_duplicates(subset=['article_id'])
articles = articles.fillna("Unknown")

# 3. Transformation
print("Transforming...")
customer_features = transactions.groupby("customer_id").size().reset_index(name="total_purchases")

# 4. Save to processed
print("Saving...")
transactions.to_parquet(f"{processed_path}/transactions_processed.parquet", index=False)
customers.to_parquet(f"{processed_path}/customers_processed.parquet", index=False)
articles.to_parquet(f"{processed_path}/articles_processed.parquet", index=False)
customer_features.to_parquet(f"{processed_path}/customer_features.parquet", index=False)

# Also save csv for easy check
transactions.to_csv(f"{processed_path}/transactions_processed.csv", index=False)

print("SUCCESS! Pipeline Completed. Check data/processed/")