# Model Card: SMS Spam Binary Classifier

## 1. Model Details

- **Model Name**: SMS Spam Binary Classifier (v1.0)
- **Model Type**: XGBoost Classifier
- **Task**: Binary classification (spam vs. ham)
- **Input**: Raw SMS text message (English)
- **Output**: Label (`spam` or `ham`)
- **Decision Threshold**: Optimized to achieve ≥76% recall and ≥89.8% precision
- **Features Used**:
  - `char_count`: Total characters
  - `word_count`: Number of words
  - `cap_ratio`: Proportion of uppercase letters
  - `exclamation_count`: Number of `!` characters
  - `question_count`: Number of `?` characters
  - `has_money`: Binary flag for money-related terms (`$`, `cash`, etc.)
  - `is_spammy_keyword`: Binary flag for known spam keywords (`free`, `win`, `urgent`, etc.)

## 2. Intended Use

- **Primary Use Case**: Filtering incoming SMS messages in a mobile or messaging application to protect users from phishing, scams, and promotional spam.
- **Users**: End-users of messaging platforms; security teams.
- **Out-of-Scope Uses**:
  - Non-English SMS messages
  - Email or social media content
  - Legal or forensic analysis (not validated for admissibility)

## 3. Training Data

- **Source**: [SMS Spam Collection Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Size**: 5,572 messages
- **Class Distribution**:
  - Spam: 747 (13.4%)
  - Ham: 4,825 (86.6%)
- **Preprocessing**:
  - Text lowercased and whitespace-normalized
  - No stemming or lemmatization
  - No removal of stop words or punctuation (punctuation is used as a signal)
- **Split**: Pre-split train/test sets (exact split not randomized in current pipeline)

## 4. Evaluation Metrics

Evaluated on held-out test set:

| Metric        | Value   | Target Met? |
|---------------|---------|-------------|
| **Recall (Spam)** | 76%     | ✅ Yes (≥76%) |
| **Precision**     | 89.8%   | ✅ Yes (≥89%) |
| **F1-Score**      | ~0.82   | — |
| **False Positive Rate** | ~10.2% | ✅ <15% tolerance |

> 💡 **Business Impact**:  
> - Catches **76% of all spam** (↑18 pts over legacy system)  
> - Only **1 in 10 messages in spam folder is legitimate** → low user frustration

## 5. Limitations

- **Language Bias**: Trained only on English-language SMS. Performance may degrade on multilingual or non-English messages.
- **Temporal Drift**: Spam tactics evolve (e.g., emoji-based spam, URL shorteners, character substitution like “fr33”). Model may degrade over time without retraining.
- **Feature Simplicity**: Relies on hand-crafted features; may miss semantic nuances that deep learning models could capture.
- **No Context**: Treats each message in isolation (no conversation history).

## 6. Ethical Considerations

- **False Positives**: Legitimate messages (e.g., from friends saying “You won the game!”) may be flagged as spam. This could cause user frustration or missed important info.
- **Fairness**: No evaluation by demographic groups (e.g., age, region). If spam patterns differ across user segments, performance may vary.
- **Transparency**: Model is interpretable via feature importance (e.g., high `cap_ratio` and `is_spammy_keyword` strongly indicate spam).

## 7. Maintenance Plan

- **Monitoring**: Track:
  - percantage of messages classified as spam (sudden changes → concept drift)
  - Feature distributions (e.g., mean `cap_ratio`) in production vs. training
- **Retraining**: Recommended every 6 months or when spam detection recall drops below 70%.
- **Feedback Loop**: Allow users to report false positives/negatives to improve future versions.

## 8. Contacts

- **Model Owner**: [Mohamed khaled]
- **Last Updated**: [3-10-2025]
- **Repository**: [https://github.com/khaldon/sms-binary-classification]