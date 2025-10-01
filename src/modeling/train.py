from pathlib import Path
import pickle

from loguru import logger
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

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


def train_model_LR(X_train, y_train):
    """Train a Logistic Regression model"""
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    logger.info("Logistic Regression Model trained successfully")
    return lr_model


def train_model_RF(X_train, y_train):
    """Train a Random Forest model"""
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    logger.info("Random Forest Model trained successfully")
    return rf_model


def train_model_XG(X_train, y_train):
    """Train a XGboosting"""
    xg_model = XGBClassifier(random_state=42)
    xg_model.fit(X_train, y_train)
    logger.info("Xgboosting Model trained successfully")
    return xg_model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and log metrics"""
    y_pred = model.predict(X_test)
    logger.info(
        f"\n{type(model).__name__ } Classification Report:\n{ classification_report(y_test, y_pred)}"
    )
    return y_pred


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

    lr_model = train_model_LR(X_train, y_train)
    rf_model = train_model_RF(X_train, y_train)
    xg_model = train_model_XG(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test)
    evaluate_model(rf_model, X_test, y_test)
    evaluate_model(xg_model, X_test, y_test)
    save_model(lr_model, MODELS_DIR / "Logistic_regression.pkl")
    save_model(rf_model, MODELS_DIR / "random_forest.pkl")
    save_model(xg_model, MODELS_DIR / "xg_boosting.pkl")
    logger.success("Training pipeline completed")


if __name__ == "__main__":
    main()
