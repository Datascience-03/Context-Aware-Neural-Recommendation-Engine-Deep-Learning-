import pandas as pd


def handle_missing_values(transaction_articles):
    """
    Handle missing values in the merged transaction + article dataset.
    """

    # Check missing values before handling
    missing_before = transaction_articles.isnull().sum()

    print("Missing values before handling:")
    print(missing_before[missing_before > 0])

    # Fill missing article descriptions
    transaction_articles["detail_desc"] = (
        transaction_articles["detail_desc"]
        .fillna("No description available")
    )

    # Check missing values after handling
    missing_after = transaction_articles.isnull().sum()

    print("\nMissing values after handling:")
    print(missing_after[missing_after > 0])

    return transaction_articles