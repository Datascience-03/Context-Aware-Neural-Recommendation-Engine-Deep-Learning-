import os
import pandas as pd

# Define paths for the reduced datasets and generated quality report.
BASE_PATH = os.path.join("data", "reduced")
REPORT_PATH = os.path.join("outputs", "reports", "data_quality_report.txt")


def load_datasets():
    articles = pd.read_csv(
        os.path.join(BASE_PATH, "articles_reduced.csv")
    )
    customers = pd.read_csv(
        os.path.join(BASE_PATH, "customers_reduced.csv")
    )
    transactions = pd.read_csv(
        os.path.join(BASE_PATH, "transactions_reduced.csv")
    )

    return articles, customers, transactions


def analyze_dataset(name, df, id_column=None):
    report = []

    report.append("")
    report.append("=" * 60)
    report.append(f"{name.upper()} DATA QUALITY")
    report.append("=" * 60)

    report.append(f"Rows: {len(df):,}")
    report.append(f"Columns: {len(df.columns)}")
    report.append(f"Duplicate rows: {df.duplicated().sum():,}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    report.append("")
    report.append("Missing values:")

    if missing.empty:
        report.append("None")
    else:
        for column, count in missing.items():
            percentage = (count / len(df)) * 100
            report.append(
                f"{column}: {count:,} ({percentage:.2f}%)"
            )

    if id_column and id_column in df.columns:
        unique_ids = df[id_column].nunique()
        duplicate_ids = df[id_column].duplicated().sum()

        report.append("")
        report.append(f"Unique {id_column}: {unique_ids:,}")
        report.append(f"Duplicate {id_column}: {duplicate_ids:,}")

    return report


def validate_customers(customers):
    report = []

    report.append("")
    report.append("=" * 60)
    report.append("CUSTOMER VALIDATION")
    report.append("=" * 60)

    if "age" in customers.columns:
        invalid_age = customers[
            customers["age"].notna()
            & (
                (customers["age"] < 10)
                | (customers["age"] > 100)
            )
        ]

        report.append(
            f"Invalid customer ages: {len(invalid_age):,}"
        )

    return report


def validate_articles(articles):
    report = []

    report.append("")
    report.append("=" * 60)
    report.append("ARTICLE VALIDATION")
    report.append("=" * 60)

    if "article_id" in articles.columns:
        invalid_article_ids = articles[
            articles["article_id"].isna()
        ]

        report.append(
            f"Missing article IDs: {len(invalid_article_ids):,}"
        )

    if "detail_desc" in articles.columns:
        missing_descriptions = articles["detail_desc"].isna().sum()

        report.append(
            f"Missing article descriptions: {missing_descriptions:,}"
        )

    return report


def validate_transactions(transactions, customers, articles):
    report = []

    report.append("")
    report.append("=" * 60)
    report.append("TRANSACTION VALIDATION")
    report.append("=" * 60)

    transaction_dates = pd.to_datetime(
        transactions["t_dat"],
        errors="coerce"
    )

    invalid_dates = transaction_dates.isna().sum()

    invalid_prices = (
        transactions["price"].isna()
        | (transactions["price"] <= 0)
    ).sum()

    valid_channels = {1, 2}

    invalid_channels = (
        ~transactions["sales_channel_id"].isin(valid_channels)
    ).sum()

    customer_ids = set(
        customers["customer_id"].dropna()
    )

    article_ids = set(
        articles["article_id"].dropna()
    )

    unknown_customers = (
        ~transactions["customer_id"].isin(customer_ids)
    ).sum()

    unknown_articles = (
        ~transactions["article_id"].isin(article_ids)
    ).sum()

    missing_customer_ids = (
        transactions["customer_id"].isna()
    ).sum()

    missing_article_ids = (
        transactions["article_id"].isna()
    ).sum()

    report.append(
        f"Invalid transaction dates: {invalid_dates:,}"
    )
    report.append(
        f"Invalid transaction prices: {invalid_prices:,}"
    )
    report.append(
        f"Invalid sales channel IDs: {invalid_channels:,}"
    )
    report.append(
        f"Missing transaction customer IDs: {missing_customer_ids:,}"
    )
    report.append(
        f"Missing transaction article IDs: {missing_article_ids:,}"
    )
    report.append(
        f"Transactions with unknown customers: {unknown_customers:,}"
    )
    report.append(
        f"Transactions with unknown articles: {unknown_articles:,}"
    )

    return report


def generate_report():
    print("Loading reduced datasets...")

    articles, customers, transactions = load_datasets()

    report = []

    report.append(
        "CONTEXT-AWARE NEURAL RECOMMENDATION ENGINE"
    )
    report.append(
        "WEEK 1 DATA QUALITY REPORT"
    )
    report.append("=" * 60)

    report.extend(
        analyze_dataset(
            "Articles",
            articles,
            "article_id"
        )
    )

    report.extend(
        analyze_dataset(
            "Customers",
            customers,
            "customer_id"
        )
    )

    report.extend(
        analyze_dataset(
            "Transactions",
            transactions
        )
    )

    report.extend(
        validate_customers(customers)
    )

    report.extend(
        validate_articles(articles)
    )

    report.extend(
        validate_transactions(
            transactions,
            customers,
            articles
        )
    )

    os.makedirs(
        os.path.dirname(REPORT_PATH),
        exist_ok=True
    )

    report_text = "\n".join(report)

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report_text)

    print(report_text)
    print("")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    generate_report()