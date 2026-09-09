from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VOCAB_FILE = PROJECT_ROOT / "data" / "processed" / "vocabularies.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "user_embedding_config.json"


USER_CATEGORICAL_FEATURES = [
    "customer_id",
    "club_member_status",
    "fashion_news_frequency",
    "postal_code",
]


def calculate_embedding_dimension(vocab_size):
    """
    Determine a practical embedding dimension from vocabulary size.

    The dimension is capped to keep the Query Tower compact.
    """

    if vocab_size <= 2:
        return 1

    if vocab_size <= 10:
        return 4

    if vocab_size <= 100:
        return 8

    if vocab_size <= 1000:
        return 16

    if vocab_size <= 10000:
        return 32

    return 64


def build_user_embedding_config():
    if not VOCAB_FILE.exists():
        raise FileNotFoundError(
            f"Vocabulary file not found: {VOCAB_FILE}"
        )

    with open(VOCAB_FILE, "r", encoding="utf-8") as file:
        vocabularies = json.load(file)

    config = {}

    for feature in USER_CATEGORICAL_FEATURES:

        if feature not in vocabularies:
            raise KeyError(
                f"Missing vocabulary for feature: {feature}"
            )

        vocabulary = vocabularies[feature]

        if not isinstance(vocabulary, dict):
            raise TypeError(
                f"Vocabulary for {feature} must be a dictionary."
            )

        vocabulary_size = len(vocabulary)

        embedding_dimension = calculate_embedding_dimension(
            vocabulary_size
        )

        config[feature] = {
            "vocabulary_size": vocabulary_size,
            "embedding_dimension": embedding_dimension,
            "padding_or_unknown_index": 0
        }

    return config


def main():

    print("=" * 60)
    print("USER EMBEDDING CONFIGURATION")
    print("=" * 60)

    config = build_user_embedding_config()

    print("\nUser embedding configuration:")

    for feature, settings in config.items():
        print(
            f"  {feature}: "
            f"vocab_size={settings['vocabulary_size']}, "
            f"embedding_dim={settings['embedding_dimension']}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            config,
            file,
            indent=4
        )

    print("\nSaved configuration to:")
    print(OUTPUT_FILE)

    print("\nUSER EMBEDDING CONFIGURATION COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()