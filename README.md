# Credit Scoring — Stacking Ensemble

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
│   ├── preprocessing/    # Data cleaning, SMOTE, splits
│   ├── features/         # Feature engineering
│   ├── models/           # Base + meta learners
│   ├── explain/          # SHAP analysis
│   └── api/              # FastAPI inference endpoint
├── tests/
└── main.py
```

## How to run

```bash
pip install -r requirements.txt
python main.py train          # Train the stack
python main.py serve          # Run FastAPI on :8000
```

Inference example:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"income": 5000, "age": 35, "credit_utilization": 0.4, ...}'
```

## Notes

The dataset is public (Kaggle) and the target definition follows the original competition. The model architecture is appropriate for the problem class but is not necessarily what I would deploy in production — choices like 4-base-learner stacking are heavier than typical fintech production scoring (usually a single well-tuned LGBM or XGBoost with proper monitoring). Stacking is used here as an exploration of the technique.

## Status

Study repository. Training and inference both work end-to-end.

## Author

Murillo Sezerino — Analytics Engineer · Data Engineer
[murillosezerino.com](https://murillosezerino.com) · [LinkedIn](https://linkedin.com/in/murillosezerino)
