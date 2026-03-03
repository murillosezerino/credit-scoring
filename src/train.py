"""
train.py — Stacking Ensemble com XGBoost, LightGBM, CatBoost e Regressao Logistica.
Meta-learner: Regressao Logistica com otimizacao Bayesiana.
"""

import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


def build_stacking_model():
    base_learners = [
        ("xgb", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )),
        ("lgbm", LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )),
        ("catboost", CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            random_state=42,
            verbose=0,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        )),
    ]

    meta_learner = LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42,
    )

    model = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )

    return model


def train(X_train, y_train) -> StackingClassifier:
    logger.info("Treinando Stacking Ensemble...")
    logger.info(f"  Base learners: XGBoost, LightGBM, CatBoost, RandomForest")
    logger.info(f"  Meta-learner:  Logistic Regression")
    logger.info(f"  Amostras de treino: {len(X_train):,}")

    model = build_stacking_model()
    model.fit(X_train, y_train)

    logger.info("Treinamento concluido.")
    return model


def save_model(model, scaler, path_model: str = "models/stacking_model.pkl",
               path_scaler: str = "models/scaler.pkl"):
    joblib.dump(model, path_model)
    joblib.dump(scaler, path_scaler)
    logger.info(f"Modelo salvo: {path_model}")
    logger.info(f"Scaler salvo: {path_scaler}")


def load_model(path_model: str = "models/stacking_model.pkl",
               path_scaler: str = "models/scaler.pkl"):
    model = joblib.load(path_model)
    scaler = joblib.load(path_scaler)
    return model, scaler