import pandas as pd


ITEM_COLUMNS = [
    "article_id",
    "product_code",
    "prod_name",
    "product_type_name",
    "product_group_name",
    "graphical_appearance_name",
    "colour_group_name",
    "department_name",
    "index_name",
]


def load_item_data(path: str) -> pd.DataFrame:
    """Load article data and prepare item-context features."""
    df = pd.read_csv(path)

    missing_columns = [col for col in ITEM_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing item columns: {missing_columns}")

    df = df[ITEM_COLUMNS].copy()

    # Fill missing categorical values
    categorical_columns = ITEM_COLUMNS[2:]
    df[categorical_columns] = df[categorical_columns].fillna("unknown")

    # Convert IDs to strings for categorical embedding layers
    df["article_id"] = df["article_id"].astype(str)
    df["product_code"] = df["product_code"].astype(str)

    return df