# Credit Scoring — Stacking Ensemble

[![CI](https://github.com/murillosezerino/credit-scoring/actions/workflows/ci.yml/badge.svg)](https://github.com/murillosezerino/credit-scoring/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009485)

> Technical study: default prediction model using a Stacking Ensemble of gradient boosting algorithms, served through a FastAPI endpoint.

A focused exercise in applied machine learning for credit risk — a domain familiar from prior experience at PagBank, Porto Vale Consórcios and CooperJohnson. The project explores ensemble methods (XGBoost + LightGBM + CatBoost + Random Forest) with a Logistic Regression meta-learner, on a public Kaggle dataset.

## What this project explores

- **Stacking Ensemble** combining four gradient boosting / tree-based base learners
- **Class imbalance handling** with SMOTE
- **Feature engineering** for credit-specific signals (TotalLatePayments, CreditUtilizationRisk)
- **Model explainability** with SHAP values
- **REST API delivery** via FastAPI for inference

## Stack

`Python` · `XGBoost` · `LightGBM` · `CatBoost` · `Scikit-Learn` · `SHAP` · `FastAPI` · `pytest`

## Results on the working dataset

| Metric | Value |
|---|---|
| AUC-ROC | ~0.80 |
| Records | 150,000 (Kaggle) |
| Base learners | 4 (XGB, LGBM, CatBoost, RF) |
| Meta-learner | Logistic Regression |

Numbers refer to the Kaggle dataset used during the study; production results will differ depending on the data distribution and target definition.

## Architecture

```
raw data → preprocessing → SMOTE → feature engineering
                                         ↓
                              base learners (XGB, LGBM, CatBoost, RF)
                                         ↓
                                 meta-learner (LogReg)
                                         ↓
                              SHAP explanations + FastAPI
```

## What's inside

```
credit-scoring/
├── src/
│   ├── preprocess.py    # Cleaning, SMOTE, feature engineering, splits
│   ├── train.py         # Base learners + meta-learner, model persistence
│   ├── evaluate.py      # Metrics, ROC curve, SHAP explanations
│   └── api.py           # FastAPI inference endpoint (/predict, /health)
├── tests/               # test_preprocess.py, test_api.py
├── main.py              # Pipeline: preprocess -> train -> evaluate -> save
└── Dockerfile           # Container for the API
```

## How to run

```bash
pip install -r requirements.txt
python main.py                                  # train + evaluate + save model
uvicorn src.api:app --host 0.0.0.0 --port 8000  # serve the API
```

Open the interactive docs at `http://localhost:8000/docs` (Swagger UI).

Inference example (10 features from the Kaggle "Give Me Some Credit" dataset):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "RevolvingUtilizationOfUnsecuredLines": 0.45,
    "age": 35,
    "NumberOfTime30_59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.32,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 6,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60_89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1
  }'
```

Response:

```json
{ "default_probability": 0.08, "risk_score": 920, "risk_label": "Baixo Risco", "recommendation": "Aprovado" }
```

## Deploy (live demo)

The API is containerized. To publish a live demo:

```bash
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api      # Swagger at /docs
```

One-click deploy on [Render](https://render.com) via the included `render.yaml` (free tier, Docker runtime). The `/`, `/health` and `/docs` routes work immediately; for a live `/predict`, commit a trained `models/stacking_model.pkl` (produced by `python main.py`) so the container ships with the model.

## Notes

The dataset is public (Kaggle) and the target definition follows the original competition. The model architecture is appropriate for the problem class but is not necessarily what I would deploy in production — choices like 4-base-learner stacking are heavier than typical fintech production scoring (usually a single well-tuned LGBM or XGBoost with proper monitoring). Stacking is used here as an exploration of the technique.

## Status

Study repository. Training and inference both work end-to-end.

## Author

Murillo Sezerino — Data Engineer & Analytics
[murillosezerino.com](https://murillosezerino.com) · [LinkedIn](https://linkedin.com/in/murillosezerino)
