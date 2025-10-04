from pathlib import Path
import pickle
from typing import Optional

from loguru import logger
import pandas as pd
import typer

from src.config import MODELS_DIR
from src.dataset import clean_text, feature_engineering
from src.modeling.train import FEATURE_COLUMNS


def load_model(model_path: Optional[Path] = None):
    """Load trained model from disk"""
    model_path = model_path or (MODELS_DIR / "xg_boosting.pkl")
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


def predict_spam(message: str, model=None, threshold=None):
    """Predict if a message is spam or ham"""
    if model is None:
        model = load_model()
    if threshold is None:
        threshold = float(open(MODELS_DIR / "threshold.txt").read())

    X = preprocess_message(message)

    prob = model.predict_proba(X)[0, 1]
    prediction = 1 if prob >= threshold else 0
    return "spam" if prediction == 1 else "ham"


app = typer.Typer()

# spam_text = "APPROVED: You qualify for $50,000 loan at 1% interest! No credit check. Apply now: http://instant-loans-usa.com"
spam_text = "Free iPhone! Click now!!!"


@app.command()
def main(
    message: str = typer.Option(spam_text, "--message", "-m", help="This optional")
):
    """Classify a single SMS message as spam or ham"""
    result = predict_spam(message=message)
    typer.echo(f"Result: {result} and the SMS message is {message}")


if __name__ == "__main__":
    app()
