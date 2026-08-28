import json
import os
import pandas as pd


class FeatureVocabularies:
    """
    Maintains and builds string-to-integer vocabulary indices for categoricals
    to be used in embeddings layer mapping.
    """

    def __init__(self):
        self.vocabularies = {}

    def fit_column(self, df: pd.DataFrame, column: str):
        """
        Builds a vocabulary mapping for a single column.
        0 is reserved for unknown/padding values.
        """
        # Convert values to strings, drop nulls, and sort for determinism
        unique_vals = sorted(df[column].dropna().unique().astype(str).tolist())
        
        # Build mapping: 1-indexed (0 reserved for UNK)
        vocab = {val: idx + 1 for idx, val in enumerate(unique_vals)}
        vocab["<UNK>"] = 0
        self.vocabularies[column] = vocab
        print(f"Built vocabulary for '{column}' with size {len(vocab)}")

    def fit_all(self, customers_df: pd.DataFrame, articles_df: pd.DataFrame):
        """
        Fits vocabularies for all categorical features in customer and articles tables.
        """
        customer_cols = ["customer_id", "club_member_status", "fashion_news_frequency", "postal_code"]
        article_cols = [
            "article_id", "prod_name", "product_type_name", 
            "product_group_name", "graphical_appearance_name", 
            "colour_group_name", "department_name", "index_name"
        ]

        print("Fitting customer vocabularies...")
        for col in customer_cols:
            if col in customers_df.columns:
                self.fit_column(customers_df, col)

        print("Fitting article vocabularies...")
        for col in article_cols:
            if col in articles_df.columns:
                self.fit_column(articles_df, col)

    def transform_column(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Transforms a column to integer indices based on the built vocabulary.
        Unseen or NaN values map to 0 (<UNK>).
        """
        df = df.copy()
        if column not in self.vocabularies:
            raise ValueError(f"Vocabulary for column '{column}' has not been fitted.")

        vocab = self.vocabularies[column]
        unk_val = vocab["<UNK>"]
        
        # Map values to vocabulary index, fallback to 0 (UNK)
        df[f"{column}_idx"] = df[column].astype(str).map(vocab).fillna(unk_val).astype(int)
        return df

    def transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Transforms multiple columns to their corresponding vocabulary indices.
        """
        df_transformed = df.copy()
        for col in columns:
            if col in df_transformed.columns and col in self.vocabularies:
                df_transformed = self.transform_column(df_transformed, col)
        return df_transformed

    def save(self, filepath: str):
        """
        Saves the fitted vocabularies to a JSON file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.vocabularies, f, indent=4)
        print(f"Vocabularies saved successfully to: {filepath}")

    def load(self, filepath: str):
        """
        Loads vocabularies from a JSON file.
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            self.vocabularies = json.load(f)
        print(f"Vocabularies loaded successfully from: {filepath}")


# -------------------------------------------------------------
# Example Usage / Test
# -------------------------------------------------------------
if __name__ == "__main__":
    custs = pd.DataFrame({
        'customer_id': ['C1', 'C2', 'C3'],
        'club_member_status': ['ACTIVE', 'ACTIVE', 'LEFT CLUB']
    })
    arts = pd.DataFrame({
        'article_id': [10001, 10002, 10003],
        'prod_name': ['T-shirt', 'Jeans', 'Dress']
    })

    vocabs = FeatureVocabularies()
    vocabs.fit_all(custs, arts)
    
    # Save/load
    test_json = "data/processed/vocabularies.json"
    vocabs.save(test_json)
    
    new_vocabs = FeatureVocabularies()
    new_vocabs.load(test_json)
    
    # Test transform
    test_df = pd.DataFrame({
        'customer_id': ['C1', 'C4'], # C4 is unseen
        'club_member_status': ['ACTIVE', None]
    })
    transformed = new_vocabs.transform(test_df, ['customer_id', 'club_member_status'])
    print(transformed)
    
    # Clean up test file
    if os.path.exists(test_json):
        os.remove(test_json)
