import os
import sys
import shutil
from pathlib import Path

# ============================================================
# WINDOWS JAVA CONFIGURATION
# ============================================================

JAVA_HOME_PATH = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"

if os.path.exists(JAVA_HOME_PATH):
    os.environ["JAVA_HOME"] = JAVA_HOME_PATH
    os.environ["PATH"] = os.path.join(JAVA_HOME_PATH, "bin") + os.pathsep + os.environ.get("PATH", "")
else:
    print("WARNING: JAVA_HOME path not found:")
    print(JAVA_HOME_PATH)
# ============================================================
# WINDOWS HADOOP CONFIGURATION
# ============================================================

HADOOP_HOME_PATH = r"C:\hadoop"

os.environ["HADOOP_HOME"] = HADOOP_HOME_PATH
os.environ["hadoop.home.dir"] = HADOOP_HOME_PATH
os.environ["PATH"] = os.path.join(HADOOP_HOME_PATH, "bin") + os.pathsep + os.environ.get("PATH", "")

# ============================================================
# PYSPARK IMPORT
# ============================================================

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType

    PYSPARK_AVAILABLE = True

except ImportError as e:
    PYSPARK_AVAILABLE = False
    print("PySpark import failed:")
    print(e)


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_PATH = PROJECT_ROOT / "data" / "reduced"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"


# ============================================================
# HELPER FUNCTION
# ============================================================

def remove_path(path):
    """
    Remove an existing file or directory.
    """

    path = Path(path)

    if path.exists():

        try:

            if path.is_dir():
                shutil.rmtree(path)

            else:
                path.unlink()

            print(f"Removed old output: {path}")

        except PermissionError:

            print(f"WARNING: Could not remove: {path}")
            print("It may still be used by another process.")


# ============================================================
# CREATE SPARK SESSION
# ============================================================

def create_spark_session():

    print()
    print("=" * 60)
    print("STARTING PYSPARK")
    print("=" * 60)

    print(f"JAVA_HOME: {os.environ.get('JAVA_HOME')}")

    spark = (
        SparkSession.builder
        .appName("ContextAwareRecommendation")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print()
    print("PYSPARK STARTED SUCCESSFULLY")
    print(f"Spark version: {spark.version}")
    print()

    return spark


# ============================================================
# MAIN PYSPARK PIPELINE
# ============================================================

def run_pyspark_pipeline():

    print("=" * 60)
    print("CONTEXT-AWARE NEURAL RECOMMENDATION ENGINE")
    print("PYSPARK DATA PROCESSING PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Check input directory
    # --------------------------------------------------------

    print()
    print("Checking input data...")

    print(f"Input directory: {BASE_PATH}")

    if not BASE_PATH.exists():

        raise FileNotFoundError(
            f"Input directory does not exist:\n{BASE_PATH}"
        )

    required_files = [
        "articles_reduced.csv",
        "customers_reduced.csv",
        "transactions_reduced.csv"
    ]

    for filename in required_files:

        filepath = BASE_PATH / filename

        if not filepath.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{filepath}"
            )

        print(f"Found: {filename}")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    PROCESSED_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Start Spark
    # --------------------------------------------------------

    spark = None

    try:

        spark = create_spark_session()

        # ====================================================
        # STEP 1 - LOAD DATA
        # ====================================================

        print()
        print("=" * 60)
        print("STEP 1 - LOADING DATA")
        print("=" * 60)

        articles_path = str(
            BASE_PATH / "articles_reduced.csv"
        )

        customers_path = str(
            BASE_PATH / "customers_reduced.csv"
        )

        transactions_path = str(
            BASE_PATH / "transactions_reduced.csv"
        )

        articles = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(articles_path)
        )

        customers = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(customers_path)
        )

        transactions = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .csv(transactions_path)
        )

        print(f"Articles rows: {articles.count()}")
        print(f"Customers rows: {customers.count()}")
        print(f"Transactions rows: {transactions.count()}")

        # ====================================================
        # STEP 2 - CLEAN TRANSACTIONS
        # ====================================================

        print()
        print("=" * 60)
        print("STEP 2 - CLEANING TRANSACTIONS")
        print("=" * 60)

        transactions = transactions.dropDuplicates()

        # Convert date

        transactions = transactions.withColumn(
            "t_dat",
            F.to_date(F.col("t_dat"))
        )

        # Remove invalid dates

        transactions = transactions.filter(
            F.col("t_dat").isNotNull()
        )

        # Remove invalid prices

        transactions = transactions.filter(
            F.col("price") > 0
        )

        print(
            f"Clean transactions rows: {transactions.count()}"
        )

        # ====================================================
        # STEP 3 - CLEAN CUSTOMERS
        # ====================================================

        print()
        print("=" * 60)
        print("STEP 3 - CLEANING CUSTOMERS")
        print("=" * 60)

        customers = customers.dropDuplicates(
            ["customer_id"]
        )

        # Fill missing age

        customers = customers.withColumn(
            "age",
            F.coalesce(
                F.col("age"),
                F.lit(30)
            )
        )

        # Clean membership status

        customers = customers.withColumn(
            "club_member_status",
            F.trim(
                F.col("club_member_status")
                .cast(StringType())
            )
        )

        print(
            f"Clean customers rows: {customers.count()}"
        )

        # ====================================================
        # STEP 4 - CLEAN ARTICLES
        # ====================================================

        print()
        print("=" * 60)
        print("STEP 4 - CLEANING ARTICLES")
        print("=" * 60)

        articles = articles.dropDuplicates(
            ["article_id"]
        )

        articles = articles.na.fill(
            "Unknown"
        )

        print(
            f"Clean articles rows: {articles.count()}"
        )

        # ====================================================
        # STEP 5 - CUSTOMER FEATURES
        # ====================================================

        print()
        print("=" * 60)
        print("STEP 5 - CUSTOMER PURCHASE FEATURES")
        print("=" * 60)

        customer_features = (
            transactions
            .groupBy("customer_id")
            .agg(
                F.count("*").alias(
                    "total_purchases"
                )
            )
        )

        print(
            f"Customer feature rows: {customer_features.count()}"
        )

        # ====================================================
        # STEP 6 - REMOVE OLD OUTPUTS
        # ====================================================

        print()
        print("=" * 60)
        print("STEP 6 - CLEANING OLD OUTPUTS")
        print("=" * 60)

        output_paths = [

            PROCESSED_PATH /
            "transactions_processed.parquet",

            PROCESSED_PATH /
            "customers_processed.parquet",

            PROCESSED_PATH /
            "articles_processed.parquet",

            PROCESSED_PATH /
            "customer_features.parquet",

            PROCESSED_PATH /
            "transactions_processed_csv"

        ]

        for path in output_paths:

            remove_path(path)

        # ====================================================
        # STEP 7 - SAVE PARQUET
        # ====================================================

        print()
        print("=" * 60)
        print("STEP 7 - SAVING PARQUET FILES")
        print("=" * 60)

        transactions.write \
            .mode("overwrite") \
            .parquet(
                str(
                    PROCESSED_PATH /
                    "transactions_processed.parquet"
                )
            )

        customers.write \
            .mode("overwrite") \
            .parquet(
                str(
                    PROCESSED_PATH /
                    "customers_processed.parquet"
                )
            )

        articles.write \
            .mode("overwrite") \
            .parquet(
                str(
                    PROCESSED_PATH /
                    "articles_processed.parquet"
                )
            )

        customer_features.write \
            .mode("overwrite") \
            .parquet(
                str(
                    PROCESSED_PATH /
                    "customer_features.parquet"
                )
            )

        print("Parquet files saved successfully.")

        # ====================================================
        # STEP 8 - SAVE TRANSACTIONS CSV
        # ====================================================

        print()
        print("=" * 60)
        print("STEP 8 - SAVING TRANSACTIONS CSV")
        print("=" * 60)

        csv_output = (
            PROCESSED_PATH /
            "transactions_processed_csv"
        )

        remove_path(csv_output)

        (
            transactions
            .coalesce(1)
            .write
            .mode("overwrite")
            .option("header", True)
            .csv(str(csv_output))
        )

        print(
            f"CSV output saved to: {csv_output}"
        )

        # ====================================================
        # STEP 9 - VALIDATION
        # ====================================================

        print()
        print("=" * 60)
        print("STEP 9 - VALIDATING OUTPUT")
        print("=" * 60)

        print(
            "Transactions:",
            transactions.count()
        )

        print(
            "Customers:",
            customers.count()
        )

        print(
            "Articles:",
            articles.count()
        )

        print(
            "Customer Features:",
            customer_features.count()
        )

        # Show sample

        print()
        print("Sample transactions:")

        transactions.show(
            5,
            truncate=True
        )

        print()
        print("Sample customer features:")

        customer_features.show(
            5,
            truncate=True
        )

        # ====================================================
        # SUCCESS
        # ====================================================

        print()
        print("=" * 60)
        print("PYSPARK PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print()
        print("Generated files:")

        print(
            PROCESSED_PATH /
            "transactions_processed.parquet"
        )

        print(
            PROCESSED_PATH /
            "customers_processed.parquet"
        )

        print(
            PROCESSED_PATH /
            "articles_processed.parquet"
        )

        print(
            PROCESSED_PATH /
            "customer_features.parquet"
        )

        print()
        print("Actual PySpark processing is COMPLETE.")
        print()

        return True

    except Exception as e:

        print()
        print("=" * 60)
        print("PYSPARK PIPELINE FAILED")
        print("=" * 60)

        print()
        print("Error:")
        print(type(e).__name__)

        print(str(e))

        print()
        print("PySpark did NOT complete successfully.")

        return False

    finally:

        if spark is not None:

            print()
            print("Stopping SparkSession...")

            try:
                spark.stop()

            except Exception:
                pass

            print("SparkSession stopped.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not PYSPARK_AVAILABLE:

        print(
            "ERROR: PySpark is not installed in this environment."
        )

        sys.exit(1)

    success = run_pyspark_pipeline()

    if success:

        sys.exit(0)

    else:

        sys.exit(1)