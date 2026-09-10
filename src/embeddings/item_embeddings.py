import pandas as pd
import tensorflow as tf

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
def encode_item_categories(df: pd.DataFrame):
    """Encode categorical item features as integer IDs."""

    encoded_df = df.copy()
    encoders = {}

    categorical_columns = ITEM_COLUMNS[2:]

    for column in categorical_columns:
        categories = sorted(encoded_df[column].astype(str).unique())

        encoder = {value: index + 1 for index, value in enumerate(categories)}

        encoded_df[column] = (
            encoded_df[column]
            .astype(str)
            .map(encoder)
            .fillna(0)
            .astype(int)
        )

        encoders[column] = encoder

    return encoded_df, encoders
<<<<<<< HEAD
def create_item_embeddings(encoded_df: pd.DataFrame, encoders: dict, embedding_dim: int = 16):
    """Create embedding layers for categorical item features."""

    embeddings = {}

    categorical_columns = ITEM_COLUMNS[2:]

    for column in categorical_columns:
        vocab_size = len(encoders[column]) + 1

        embedding_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        )

        embeddings[column] = embedding_layer

    return embeddings
=======
>>>>>>> origin/feature/member-4-contextual-features
