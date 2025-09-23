# src/dataset.py
"""
SMS Spam Classification — Preprocessing Pipeline
Applies cleaning & feature engineering to pre-split train/test sets.
"""

from pathlib import Path
import re
from typing import Tuple

from loguru import logger
import pandas as pd
import typer

from src.config import PROCESSED_DATA_DIR

# Default filenames (adjust if yours are different)
DEFAULT_TRAIN_FILENAME = "train_set.csv"
DEFAULT_TEST_FILENAME = "test_set.csv"


# ====================
# 1. LOAD TRAIN DATA
# ====================
def load_train_data(input_path) -> pd.DataFrame:
    """Load pre-split train set."""
    input_path = input_path or (PROCESSED_DATA_DIR / DEFAULT_TRAIN_FILENAME)
    logger.info(f"📥 Loading train set from {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"❌ Train file not found: {input_path}")

    df = pd.read_csv(input_path)

    # Validate required columns
    required = {"label", "message"}
    missing = required - set(df.columns)
    assert not missing, f"❌ Missing columns in train set: {missing}"
    assert df["label"].isin(["spam", "ham"]).all(), "❌ Invalid labels in train set"
    assert not df.isnull().any().any(), "❌ Missing values in train set"

    spam_ratio = df["label"].value_counts(normalize=True).get("spam", 0)
    logger.info(f"✅ Loaded train set: {len(df):,} rows. Spam ratio: {spam_ratio:.2%}")
    return df


# ====================
# 2. LOAD TEST DATA
# ====================
def load_test_data(test_path) -> pd.DataFrame:
    """Load pre-split test set."""
    test_path = test_path or (PROCESSED_DATA_DIR / DEFAULT_TEST_FILENAME)
    logger.info(f"📥 Loading test set from {test_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"❌ Test file not found: {test_path}")

    df = pd.read_csv(test_path)

    required = {"label", "message"}
    missing = required - set(df.columns)
    assert not missing, f"❌ Missing columns in test set: {missing}"
    assert df["label"].isin(["spam", "ham"]).all(), "❌ Invalid labels in test set"
    assert not df.isnull().any().any(), "❌ Missing values in test set"

    logger.info(f"✅ Loaded test set: {len(df):,} rows")
    return df


# ====================
# 3. CLEAN TEXT
# ====================
def clean_text(text: str) -> str:
    """Clean text: lowercase, remove extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)  # ⚠️ You had \s+ → "" — that removes ALL spaces!
    return text.strip()


# ====================
# 4. FEATURE ENGINEERING
# ====================
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add new features based on EDA."""
    df = df.copy()

    df["char_count"] = df["message"].str.len()
    df["word_count"] = df["message"].str.split().str.len()
    df["cap_ratio"] = df["message"].apply(
        lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
    )
    df["exclamation_count"] = df["message"].str.count("!")
    df["question_count"] = df["message"].str.count(r"\?")
    df["has_money"] = (
        df["message"].str.contains(r"\$|£|€|money|cash", case=False).astype(int)
    )
    df["is_spammy_keyword"] = (
        df["message"]
        .str.contains(r"free|win|prize|urgent|claim|congratulations", case=False)
        .astype(int)
    )

    logger.info("✅ Engineered 7 new features.")
    return df


# ====================
# 5. ENCODE LABELS
# ====================
def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Encode 'spam'=1, 'ham'=0."""
    df = df.copy()
    df["label_encoded"] = df["label"].map({"spam": 1, "ham": 0})
    if df["label_encoded"].isnull().any():
        raise ValueError("❌ Found unmapped labels")
    logger.info("✅ Labels encoded: 'spam'→1, 'ham'→0")
    return df


# ====================
# 6. SAVE AS PARQUET
# ====================
def save_processed_sets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_dir,
    dry_run: bool = False,
):
    """Save train and test sets as parquet files."""
    if dry_run:
        logger.info("⏭️  Dry run — skipping save.")
        return

    output_dir = output_dir or PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"

    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)

    logger.info(f"💾 Saved train set → {train_path}")
    logger.info(f"💾 Saved test set  → {test_path}")


# ====================
# 7. MAIN PIPELINE
# ====================
def run_preprocessing(
    input_path: Path = None,
    test_path: Path = None,
    output_dir: Path = None,
    dry_run: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load pre-split train/test sets → clean → engineer features → encode → save.
    """
    logger.info("🚀 Starting preprocessing pipeline...")

    # 1. Load
    df_train = load_train_data(input_path)
    df_test = load_test_data(test_path)

    # 2. Clean text
    df_train["message_clean"] = df_train["message"].apply(clean_text)
    df_test["message_clean"] = df_test["message"].apply(clean_text)

    # 3. Engineer features
    df_train = feature_engineering(df_train)
    df_test = feature_engineering(df_test)

    # 4. Encode labels
    df_train = encode_labels(df_train)
    df_test = encode_labels(df_test)

    # 5. Save
    save_processed_sets(df_train, df_test, output_dir, dry_run)

    logger.success("✅ Preprocessing completed successfully.")
    return df_train, df_test


app = typer.Typer()


@app.command()
def main(
    input_path: Path = typer.Option(
        PROCESSED_DATA_DIR / DEFAULT_TRAIN_FILENAME, "--input-path", "-i"
    ),
    test_path: Path = typer.Option(
        PROCESSED_DATA_DIR / DEFAULT_TEST_FILENAME, "--test-path", "-t"
    ),
    output_dir: Path = typer.Option(PROCESSED_DATA_DIR, "--output-dir", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """
    Run SMS spam preprocessing pipeline.
    """
    train, test = run_preprocessing(
        input_path=input_path,
        test_path=test_path,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    logger.success(f"🎉 Done! Train: {len(train):,} rows, Test: {len(test):,} rows")


# ====================
# 10. SCRIPT ENTRY
# ====================
if __name__ == "__main__":
    app()
