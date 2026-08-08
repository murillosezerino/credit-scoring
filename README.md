# Credit Scoring

[![CI](https://github.com/murillosezerino/credit-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/murillosezerino/credit-scoring/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009485)

> Technical study: default-risk prediction served through a FastAPI endpoint. A calibrated LightGBM is the serving model; a stacking ensemble is kept as an exploration of the technique.

Applied machine learning for credit risk, a domain familiar from prior experience at PagBank, Porto Vale and CooperJohnson. The repository runs end to end out of the box on a small synthetic sample, so you can clone it and get a live `/predict` in one command. Study metrics come from the public Kaggle dataset.

## Two models, on purpose

- **Serving model (what the API uses):** a single **LightGBM with calibrated probabilities** (isotonic). No SMOTE here; class imbalance is handled with class weights and probabilities are calibrated on the real class distribution, so `default_probability` is interpretable.
- **Study model:** a **Stacking Ensemble** (XGBoost + LightGBM + CatBoost + Random Forest with a Logistic Regression meta-learner), kept to explore the technique.

## What this project explores

- Correct leakage-free preprocessing: fit the scaler on the training split only, then apply **SMOTE after scaling** (so the KNN distances are meaningful and the scaler is never influenced by synthetic points)
- Probability **calibration** for a meaningful risk score
- **Feature engineering** for credit signals (TotalLatePayments, IncomePerDependent, CreditUtilizationRisk)
- **Explainability** with SHAP on the served model
- **REST API delivery** via FastAPI, containerized for a live demo

## Stack

`Python` - `LightGBM` - `XGBoost` - `CatBoost` - `Scikit-Learn` - `SHAP` - `FastAPI` - `Docker` - `pytest`

## Results

| Metric | Study (full Kaggle data) |
|---|---|
| AUC-ROC | ~0.80 |
| Records | 150,000 (Kaggle "Give Me Some Credit") |
| Base learners | 4 (XGB, LGBM, CatBoost, RF) |
| Meta-learner | Logistic Regression |

Numbers refer to the full Kaggle dataset used during the study. The out-of-the-box run uses a synthetic sample, so its metrics are not comparable and are only meant to prove the pipeline runs.

## Architecture

```
raw data -> clean -> feature engineering -> split -> fit scaler (train only) -> scale
     |                                                                 |
     |                                              +------------------+------------------+
     |                                              |                                     |
     v                                       serving model                          study model
 (Kaggle or                          LightGBM + isotonic calibration         SMOTE -> stacking ensemble
  synthetic                          -> SHAP -> models/serving_model.pkl      (XGB+LGBM+CatBoost+RF -> LogReg)
  sample)                                     |
                                             v
                                    FastAPI /predict (live)
```

## What's inside

```
credit-scoring/
├── src/
│   ├── preprocess.py    # clean, feature engineering, split_scale, apply_smote
│   ├── train.py         # serving model (calibrated LGBM) + stacking study
│   ├── evaluate.py      # metrics, ROC curve, SHAP (headless Agg backend)
│   └── api.py           # FastAPI inference endpoint (/predict, /health)
├── scripts/
│   ├── make_sample.py   # generate the synthetic sample (runs OOTB)
│   └── download_data.py # fetch the real Kaggle dataset (optional)
├── data/sample/         # committed synthetic sample
├── models/              # committed serving_model.pkl + scaler.pkl (for the demo)
├── tests/               # test_preprocess.py, test_api.py (TestClient)
├── main.py              # pipeline: preprocess -> serving -> study
└── Dockerfile
```

## How to run

Out of the box, no Kaggle account needed:

```bash
pip install -r requirements.txt
python scripts/make_sample.py                   # writes data/sample/cs-training-sample.csv
python main.py                                  # train + evaluate + save serving model
uvicorn src.api:app --host 0.0.0.0 --port 8000  # serve the API
```

Open the interactive docs at `http://localhost:8000/docs`. Set `TRAIN_STUDY=0` to skip the heavier stacking and train only the serving model.

With the real Kaggle data:

```bash
python scripts/download_data.py                 # needs Kaggle API credentials
DATA_PATH=data/cs-training.csv python main.py
```

Inference example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.45, "age": 35,
    "NumberOfTime30_59DaysPastDueNotWorse": 0, "DebtRatio": 0.32,
    "MonthlyIncome": 5000, "NumberOfOpenCreditLinesAndLoans": 6,
    "NumberOfTimes90DaysLate": 0, "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60_89DaysPastDueNotWorse": 0, "NumberOfDependents": 1
  }'
```

Response:

```json
{ "default_probability": 0.03, "risk_score": 970, "risk_label": "Baixo Risco", "recommendation": "Aprovado" }
```

## Deploy (live demo)

The serving model (`models/serving_model.pkl`) and scaler ship in the repository, so the container has a live `/predict` with no extra steps. One-click deploy on [Render](https://render.com) via the included `render.yaml` (free tier, Docker runtime).

```bash
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api      # Swagger at /docs
```

Honest note: the hosted demo model is trained on the synthetic sample so the endpoint is live without shipping the large dataset. The study metrics (AUC ~0.80) come from the full Kaggle data. Render free tier hibernates, so the first request after idle has a cold start.

## Notes

The dataset is public (Kaggle) and the target follows the original competition. The 4-base-learner stacking is heavier than typical fintech production scoring (usually a single well-tuned, calibrated LGBM with monitoring), which is exactly why the **serving** model here is a single calibrated LGBM; the stacking is kept as a study. In production the risk score would come from a proper scorecard (WOE, points to double the odds) rather than the linear transform used here.

## Status

Study repository. Preprocessing, training, serving and the API all run end to end, verified by tests.

## Author

Murillo Sezerino, Data Engineer
[murillosezerino.com](https://murillosezerino.com) - [LinkedIn](https://linkedin.com/in/murillosezerino)
