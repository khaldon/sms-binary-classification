from pathlib import Path
import pickle
from typing import List, Optional

from loguru import logger
import pandas as pd
import typer

from src.config import MODELS_DIR
from src.dataset import clean_text, feature_engineering
from src.modeling.train import FEATURE_COLUMNS


def load_model(model_path: Optional[Path] = None):
    """Load trained model from disk"""
    model_path = model_path or (MODELS_DIR / "Logistic_regression.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return model


def preprocess_message(message: str):
    """Preprocess a single message for prediction"""
    df = pd.DataFrame({"message": [message]})
    df["message_clean"] = df["message"].apply(clean_text)
    df = feature_engineering(df)
    return df[FEATURE_COLUMNS]


def predict_spam(message: str, model=None):
    """Predict if a message is spam or ham"""
    if model is None:
        model = load_model()

    X = preprocess_message(message)

    prediction = model.predict(X)[0]
    label = "spam" if prediction == 1 else "ham"
    logger.info(f"Message: '{message}' -> {label.upper()}")
    return label


app = typer.Typer()


@app.command()
def main(
    message: str = typer.Option(..., "--message", "-m", help="SMS message to classify")
):
    """Classify a single SMS message as spam or ham"""
    result = predict_spam(message)
    typer.echo(f"Result: {result}")


if __name__ == "__main__":
    app()
