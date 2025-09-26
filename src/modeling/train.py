from pathlib import Path
import pickle

from loguru import logger
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

FEATURE_COLUMNS = [
    "char_count",
    "word_count",
    "cap_ratio",
    "exclamation_count",
    "question_count",
    "has_money",
    "is_spammy_keyword",
]

TARGET_COLUMN = "label_encoded"


def load_data():
    """Load preprocessed train and test sets"""
    train = pd.read_parquet(f"{PROCESSED_DATA_DIR}/train.parquet")
    test = pd.read_parquet(f"{PROCESSED_DATA_DIR}/test.parquet")
    return train, test


def train_model(X_train, y_train):
    """Train a Logistic Regression model"""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    logger.info("Model trained successfully")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and log metrics"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    return acc, y_pred


def save_model(model, path: Path):
    """Save model to disk"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def main():
    logger.info("Starting model training...")

    train, test = load_data()

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    model = train_model(X_train, y_train)
    
    acc, y_pred = evaluate_model(model, X_test, y_test)
    save_model(model, MODELS_DIR / "Logistic_regresion.pkl")
    logger.success("Training pipeline completed")


if __name__ == "__main__":
    main()
