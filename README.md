# sms-binary-classification

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A production-ready binary classifier to detect spam SMS messages using engineered features and XGBoost. 
**Key result:** Our model catches 76% of all spam SMS messages a 18% point improvment over the current system (58% -> 76%). This significantly reduces phishing risk. At the same time Of all SMS messages flagged as spam, 89.8% are truly spam, meaning only 1 in 10 SMS in the spam folder is actually legitimate well below the 15% false alarm tolerance

### Quick Start 

#### 1. Activate the enviroment and install dependencies 
This project uses uv package manager

```bash
# create enviroment 
uv init 

# activate enviroment 
source .venv/bin/activate

# install dependencies 
uv sync 
```
#### 2. Dataset
This is the dataset used in this model to quick start:
[SMS Spam collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

you have to run first `notebooks/01_eda.ipynb`to split the dataset to trainset and testset ***just to prevent the data leakage***. 
you will find the splitting in `data/processed/`.

Each must contain:
- message: raw SMS text
- label: "spam" or "ham"


### 3.Run the full pipline 

```bash
# Preprocess data → train models → evaluate
python src/dataset.py
python src/modeling/train.py
```
or use the `Makefile`: 
```bash
make data 
make train 
```
### 4. Make a prediction

```bash
python src/modeling/predict.py
```
To classify your own message, edit the `main()` in `predict.py` you can either use CLI input or hardcoded the message. 

| Model                   | Precision   | Recall   | F1-Score   | Accuracy   |
|-------------------------|-------------|----------|------------|------------|
| MVP Baseline            | >=0.85      | >=0.75   | >=0.80     | -          |
| Dumb Baseline (All Ham) | 0.0000      | 0.0000   | 0.0000     | 0.8655     |
| Logistic Regression     | 0.7909      | 0.5800   | 0.6692     | 0.9229     |
| Random Forest           | 0.9237      | 0.7267   | 0.8134     | 0.9552     |
| **XGboosting**              | **0.8976**      | **0.7600**   | **0.8231**     | **0.9561**     |


**Meets both security (high recall) and user experience (low false positives) goals.**

Model: XGBoost (outperformed Logistic Regression and Random Forest)

Features: Text length, capitalization ratio, punctuation counts, keyword flags (free, win, urgent, etc.)

Full evaluation in:

`notebooks/evaluation_model.ipynb`


## Output Locations

| Output | Path |
|---|---|
| Processed data | `data/processed/train.parquet` ,  `test.parquet` |
| Trained models | `models/xg_boosting.pkl`, `random_forest.pkl`, `Logistic_regression.pkl` |
| Evaluation report | `reports/model_evaluation_summary.md` |
| EDA & analysis | `notebooks/` |

## Project Organization

(Standardized via Cookiecutter Data Science )

```
├── data/processed/        ← Input: train_set.csv, test_set.csv
├── models/                ← Output: .pkl model files
├── reports/               ← Model summary & evaluation
├── notebooks/             ← EDA and experimentation
└── src/
    ├── dataset.py         ← Preprocessing & feature engineering
    └── modeling/
        ├── train.py       ← Trains 3 models, saves all
        └── predict.py     ← Classifies new SMS

```
--------

