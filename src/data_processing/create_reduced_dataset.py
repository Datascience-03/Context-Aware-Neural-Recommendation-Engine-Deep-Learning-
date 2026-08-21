import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "reduced"

TRANSACTIONS_FILE = RAW_DIR / "transactions_train.csv"
CUSTOMERS_FILE = RAW_DIR / "customers.csv"
ARTICLES_FILE = RAW_DIR / "articles.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Settings
# --------------------------------------------------

TARGET_TRANSACTIONS = 200_000
CHUNK_SIZE = 50_000

print("Creating reduced H&M dataset...")
print(f"Target transactions: {TARGET_TRANSACTIONS}")

# --------------------------------------------------
# 1. Select transactions
# --------------------------------------------------

transaction_chunks = []

rows_collected = 0

for chunk in pd.read_csv(TRANSACTIONS_FILE, chunksize=CHUNK_SIZE):

    remaining = TARGET_TRANSACTIONS - rows_collected

    if remaining <= 0:
        break

    if len(chunk) > remaining:
        chunk = chunk.iloc[:remaining]

    transaction_chunks.append(chunk)

    rows_collected += len(chunk)

    print(f"Collected transactions: {rows_collected:,}")

transactions = pd.concat(transaction_chunks, ignore_index=True)

transactions_output = OUTPUT_DIR / "transactions_reduced.csv"

transactions.to_csv(
    transactions_output,
    index=False
)

print(f"\nSaved: {transactions_output}")
print(f"Transactions: {len(transactions):,}")

# --------------------------------------------------
# 2. Find required customers
# --------------------------------------------------

customer_ids = set(
    transactions["customer_id"].astype(str)
)

print(f"Unique customers required: {len(customer_ids):,}")

customer_chunks = []

for chunk in pd.read_csv(
    CUSTOMERS_FILE,
    chunksize=CHUNK_SIZE,
    dtype={"customer_id": str}
):

    matching = chunk[
        chunk["customer_id"].isin(customer_ids)
    ]

    if not matching.empty:
        customer_chunks.append(matching)

customers = pd.concat(
    customer_chunks,
    ignore_index=True
)

customers_output = OUTPUT_DIR / "customers_reduced.csv"

customers.to_csv(
    customers_output,
    index=False
)

print(f"Saved: {customers_output}")
print(f"Customers: {len(customers):,}")

# --------------------------------------------------
# 3. Find required articles
# --------------------------------------------------
print("\n[3/3] Selecting articles...")

# Normalize article IDs to 10-digit strings
# H&M article IDs in articles.csv contain leading zeros.
transactions["article_id"] = (
    transactions["article_id"]
    .astype(str)
    .str.replace(".0", "", regex=False)
    .str.zfill(10)
)

article_ids = set(
    transactions["article_id"]
    .dropna()
    .unique()
)

print(f"Unique articles required: {len(article_ids):,}")

# Read articles dataset
articles = pd.read_csv(
    ARTICLES_FILE,
    dtype={"article_id": str}
)

# Normalize article IDs
articles["article_id"] = (
    articles["article_id"]
    .astype(str)
    .str.strip()
    .str.zfill(10)
)

# Select only articles used by our reduced transactions
articles = articles[
    articles["article_id"].isin(article_ids)
].copy()

articles_output = OUTPUT_DIR / "articles_reduced.csv"

articles.to_csv(
    articles_output,
    index=False
)

print(f"Saved: {articles_output}")
print(f"Articles: {len(articles):,}")

# --------------------------------------------------
# Final Summary
# --------------------------------------------------

print("\n==============================================")
print("REDUCED DATASET CREATED SUCCESSFULLY")
print("==============================================")

print(f"Transactions : {len(transactions):,}")
print(f"Customers    : {len(customers):,}")
print(f"Articles     : {len(articles):,}")

print("\nFiles:")
print(transactions_output)
print(customers_output)
print(articles_output)

print("\nDone!")