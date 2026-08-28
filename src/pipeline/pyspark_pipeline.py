import os
import pandas as pd

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


def clean_with_pyspark(base_path, processed_path):
    """
    Cleans the datasets using PySpark.
    """
    print("Running Spark-based pipeline...")
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("HMRecommendationPipeline") \
        .master("local[*]") \
        .getOrCreate()

    try:
        # 1. Load data
        print("Loading CSV files into Spark DataFrames...")
        articles = spark.read.csv(f"{base_path}/articles_reduced.csv", header=True, inferSchema=True)
        customers = spark.read.csv(f"{base_path}/customers_reduced.csv", header=True, inferSchema=True)
        transactions = spark.read.csv(f"{base_path}/transactions_reduced.csv", header=True, inferSchema=True)

        print(f"Articles count: {articles.count()}, Customers count: {customers.count()}, Transactions count: {transactions.count()}")

        # 2. Cleaning - Member 2 Task
        print("Cleaning transactions, customers, and articles using Spark...")
        # Transactions
        transactions = transactions.dropDuplicates()
        transactions = transactions.withColumn("t_dat", F.to_date(F.col("t_dat")))
        transactions = transactions.filter(F.col("t_dat").isNotNull())
        transactions = transactions.filter(F.col("price") > 0)

        # Customers
        customers = customers.dropDuplicates(["customer_id"])
        customers = customers.withColumn("age", F.coalesce(F.col("age"), F.lit(30)))
        customers = customers.withColumn("club_member_status", F.trim(F.col("club_member_status").cast(StringType())))

        # Articles
        articles = articles.dropDuplicates(["article_id"])
        articles = articles.na.fill("Unknown")

        # 3. Transformation: total purchases per customer
        print("Transforming and calculating customer features...")
        customer_features = transactions.groupBy("customer_id").count().withColumnRenamed("count", "total_purchases")

        # 4. Save to processed in Parquet format
        print("Saving cleaned Spark DataFrames to processed parquet...")
        transactions.write.mode("overwrite").parquet(f"{processed_path}/transactions_processed.parquet")
        customers.write.mode("overwrite").parquet(f"{processed_path}/customers_processed.parquet")
        articles.write.mode("overwrite").parquet(f"{processed_path}/articles_processed.parquet")
        customer_features.write.mode("overwrite").parquet(f"{processed_path}/customer_features.parquet")

        # Save transactions to CSV as well (coalesced to 1 file for ease)
        transactions.coalesce(1).write.mode("overwrite").csv(f"{processed_path}/transactions_processed_csv", header=True)
        
        # Move CSV out of directory if needed, or keep as is
        print("Spark cleaning finished successfully!")
        return True
    finally:
        spark.stop()


def clean_with_pandas_fallback(base_path, processed_path):
    """
    Fallback Pandas cleaning function in case PySpark cannot run (e.g. Java is missing).
    """
    print("PySpark execution not possible or failed. Falling back to Pandas pipeline...")
    # 1. Load
    articles = pd.read_csv(f"{base_path}/articles_reduced.csv")
    customers = pd.read_csv(f"{base_path}/customers_reduced.csv")
    transactions = pd.read_csv(f"{base_path}/transactions_reduced.csv")

    # 2. Cleaning
    transactions = transactions.drop_duplicates()
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'], errors='coerce')
    transactions = transactions.dropna(subset=['t_dat'])
    transactions = transactions[transactions['price'] > 0]

    customers = customers.drop_duplicates(subset=['customer_id'])
    customers['age'] = customers['age'].fillna(30)
    customers['club_member_status'] = customers['club_member_status'].astype(str).str.strip()

    articles = articles.drop_duplicates(subset=['article_id'])
    articles = articles.fillna("Unknown")

    # 3. Transformation
    customer_features = transactions.groupby("customer_id").size().reset_index(name="total_purchases")

    # 4. Save
    transactions.to_parquet(f"{processed_path}/transactions_processed.parquet", index=False)
    customers.to_parquet(f"{processed_path}/customers_processed.parquet", index=False)
    articles.to_parquet(f"{processed_path}/articles_processed.parquet", index=False)
    customer_features.to_parquet(f"{processed_path}/customer_features.parquet", index=False)

    transactions.to_csv(f"{processed_path}/transactions_processed.csv", index=False)
    print("Pandas fallback pipeline completed successfully!")


def main():
    base_path = "data/reduced"
    processed_path = "data/processed"
    os.makedirs(processed_path, exist_ok=True)

    success = False
    if SPARK_AVAILABLE:
        try:
            success = clean_with_pyspark(base_path, processed_path)
        except Exception as e:
            print(f"Failed to execute PySpark pipeline due to: {e}")
            
    if not success:
        clean_with_pandas_fallback(base_path, processed_path)


if __name__ == "__main__":
    main()