import pandas as pd
import numpy as np


def get_popular_items(transactions: pd.DataFrame, limit: int = 10) -> list:
    """
    Returns the top N popular item IDs based on transaction counts.
    """
    if transactions.empty:
        return []
    popular = transactions['article_id'].value_counts().index[:limit].tolist()
    return popular


def get_demographic_popular_items(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    age: int,
    age_window: int = 5,
    limit: int = 10
) -> list:
    """
    Returns top N popular items for a demographic age bracket (age +/- age_window).
    """
    if transactions.empty or customers.empty:
        return []

    # Get customers in the age range
    min_age = age - age_window
    max_age = age + age_window
    demographic_customers = customers[
        (customers['age'] >= min_age) & (customers['age'] <= max_age)
    ]['customer_id']

    if demographic_customers.empty:
        return get_popular_items(transactions, limit=limit)

    # Filter transactions
    demographic_tx = transactions[transactions['customer_id'].isin(demographic_customers)]
    if demographic_tx.empty:
        return get_popular_items(transactions, limit=limit)

    return demographic_tx['article_id'].value_counts().index[:limit].tolist()


def handle_user_cold_start(
    customer_id: str,
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    limit: int = 10
) -> list:
    """
    Checks if a user is unseen in transactions (cold start).
    If cold start, returns popular recommendations (demographic-based if age is available, else overall).
    Otherwise returns None (can be handled by normal recommendation model).
    """
    # If customer is already active, return None (no cold start needed)
    if not transactions.empty and customer_id in transactions['customer_id'].values:
        return None

    # Get customer age if available
    customer_info = customers[customers['customer_id'] == customer_id]
    if not customer_info.empty and not pd.isna(customer_info['age'].values[0]):
        user_age = int(customer_info['age'].values[0])
        return get_demographic_popular_items(transactions, customers, age=user_age, limit=limit)

    # Fallback to general popular items
    return get_popular_items(transactions, limit=limit)


def handle_item_cold_start(
    article_id: int,
    articles: pd.DataFrame,
    transactions: pd.DataFrame,
    limit: int = 10
) -> list:
    """
    Checks if an item is unseen in transactions (cold start).
    If cold start, finds similar items in the same product group and returns the most popular ones.
    Otherwise returns None.
    """
    # If article is already in transactions, it's not a cold-start item
    if not transactions.empty and article_id in transactions['article_id'].values:
        return None

    # Find category/product group of the cold start item
    item_info = articles[articles['article_id'] == article_id]
    if item_info.empty:
        return get_popular_items(transactions, limit=limit)

    prod_group = item_info['product_group_name'].values[0]

    # Find other articles in the same product group
    similar_articles = articles[articles['product_group_name'] == prod_group]['article_id']

    if similar_articles.empty or transactions.empty:
        return get_popular_items(transactions, limit=limit)

    # Find the most popular items in that product group
    group_tx = transactions[transactions['article_id'].isin(similar_articles)]
    if group_tx.empty:
        return get_popular_items(transactions, limit=limit)

    return group_tx['article_id'].value_counts().index[:limit].tolist()


# -------------------------------------------------------------
# Example Usage / Test
# -------------------------------------------------------------
if __name__ == "__main__":
    # Test transactions
    tx_df = pd.DataFrame({
        'customer_id': ['U1', 'U1', 'U2', 'U3', 'U3', 'U3'],
        'article_id': [1001, 1002, 1001, 1003, 1003, 1001]
    })
    # Test customers
    cust_df = pd.DataFrame({
        'customer_id': ['U1', 'U2', 'U3', 'U_NEW'],
        'age': [25, 30, 26, 45]
    })
    # Test articles
    art_df = pd.DataFrame({
        'article_id': [1001, 1002, 1003, 1004],
        'product_group_name': ['Upper wear', 'Upper wear', 'Lower wear', 'Lower wear']
    })

    print("Overall popular:", get_popular_items(tx_df, limit=2))
    print("Demographic popular (age 27):", get_demographic_popular_items(tx_df, cust_df, age=27, limit=2))
    print("Cold start new user U_NEW:", handle_user_cold_start('U_NEW', tx_df, cust_df, limit=2))
    print("Cold start new item 1004:", handle_item_cold_start(1004, art_df, tx_df, limit=2))
