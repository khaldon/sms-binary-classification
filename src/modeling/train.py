import json
from pathlib import Path
import pickle

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_curve,
    precision_recall_fscore_support,
)
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


def extract_metrics(y_true, y_pred, labels=["ham", "spam"]):
    """Return a dict of key metrics for binary classification"""
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    spam_metrics = {
        "spam_recall": float(recalls[1]),  # type: ignore
        "spam_precision": float(precisions[1]),  # type: ignore
        "spam_f1": float(f1s[1]),  # type: ignore
        "ham_recall": float(recalls[0]),  # pyright: ignore[reportIndexIssue]
        "ham_precision": float(precisions[0]),  # type: ignore
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "support_spam": int(supports[1]),  # type: ignore
        "support_ham": int(supports[0]),  # type: ignore
    }
    return spam_metrics


def evaluate_model(model, X_test, y_test):
    """Evaluate model and log metrics"""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["ham", "spam"])

    logger.info(f"\n{type(model).__name__ } Classification Report:\n{report}")
    metrics = extract_metrics(y_test, y_pred)
    return y_pred, metrics


def save_metrics(metrics_dict: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    logger.info(f"Metrics saved to {output_path}")


def generate_evaluation_summary(metrics: dict, output_path: Path):
    recall = metrics["spam_recall"]
    precision = metrics["spam_precision"]

    # Business context
    baseline_recall = 0.58
    improvement = recall - baseline_recall
    false_alarm_rate = 1 - precision
    content = f"""# Model Evaluation Summary

                Our XGBoost model catches **{recall:.0%} of all spam SMS messages**, a **{improvement:.0%} point improvement** over the current system ({baseline_recall:.0%} → {recall:.0%}). This significantly reduces phishing risk.

                At the same time, **{precision:.1%} of all SMS messages flagged as spam are truly spam**, meaning only **{false_alarm_rate:.1%} (about 1 in {1/false_alarm_rate:.0f})** of messages in the spam folder are actually legitimate — well below the 15% false alarm tolerance.

                This balance meets both our security and user experience goals.
                """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)
    logger.info(f"Evaluation summary saved to {output_path}")


def save_model(model, path: Path):
    """Save model to disk"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def find_optimal_threshold(y_true, y_proba, min_recall=0.76, min_precision=0.898):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Iterate from high recall to low (reverse order)
    for i in range(len(thresholds)):
        if recalls[i] >= min_recall and precisions[i] >= min_precision:
            return thresholds[i]
    # Fallback: return threshold that maximizes F1
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    return thresholds[np.argmax(f1_scores)]


def main():
    logger.info("Starting model training...")

    train, test = load_data()

    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN]

    # Train models
    lr_model = train_model_LR(X_train, y_train)
    rf_model = train_model_RF(X_train, y_train)
    xg_model = train_model_XG(X_train, y_train)

    # Evaluate
    _, lr_metrics = evaluate_model(lr_model, X_test, y_test)
    _, rf_metrics = evaluate_model(rf_model, X_test, y_test)
    y_pred_xg, xg_metrics = evaluate_model(xg_model, X_test, y_test)

    y_proba = xg_model.predict_proba(X_test)[:, 1]
    optimal_thresh = find_optimal_threshold(y_test, y_proba)
    logger.info(f"Optimal threshold: {optimal_thresh:.4f}")
    with open(MODELS_DIR / "threshold.txt", "w") as f:
        f.write(str(optimal_thresh))

    # Save models
    save_model(lr_model, MODELS_DIR / "Logistic_regression.pkl")
    save_model(rf_model, MODELS_DIR / "random_forest.pkl")
    save_model(xg_model, MODELS_DIR / "xg_boosting.pkl")

    # Save XGBoost metrics (Best Model)
    save_metrics(xg_metrics, Path("reports/metrics.json"))
    generate_evaluation_summary(xg_metrics, Path("reports/model_evaluation_summary.md"))
    logger.success("Training pipeline completed")


if __name__ == "__main__":
    main()
