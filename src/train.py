"""
train.py — Dois modelos com propositos distintos:

- Modelo de serving (o que a API usa): um LightGBM unico com probabilidades
  CALIBRADAS (isotonic). Sem SMOTE: o desbalanceamento e tratado por class_weight
  e a calibracao e feita na distribuicao real das classes, para que a
  probabilidade de default seja interpretavel.

- Modelo de estudo: um Stacking Ensemble (XGBoost + LightGBM + CatBoost +
  RandomForest -> Logistic Regression), mantido como exploracao da tecnica.
"""

import logging
import os
import joblib
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


def _lgbm(**overrides) -> LGBMClassifier:
    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    params.update(overrides)
    return LGBMClassifier(**params)


# ── Modelo de serving (calibrado) ─────────────────────────────────────

def build_serving_model():
    """LightGBM calibrado (isotonic, cv=5) na distribuicao real."""
    return CalibratedClassifierCV(_lgbm(), method="isotonic", cv=5)


def train_serving(X_train, y_train):
    logger.info("Treinando modelo de serving (LightGBM calibrado)...")
    logger.info(f"  Amostras de treino: {len(X_train):,}")
    model = build_serving_model()
    model.fit(X_train, y_train)
    logger.info("Modelo de serving treinado.")
    return model


def train_explainer(X_train, y_train) -> LGBMClassifier:
    """LGBM base (nao calibrado) usado apenas para SHAP do modelo servido."""
    model = _lgbm()
    model.fit(X_train, y_train)
    return model


# ── Modelo de estudo (stacking) ───────────────────────────────────────

def build_stacking_model():
    base_learners = [
        ("xgb", XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )),
        ("lgbm", _lgbm(class_weight=None)),
        ("catboost", CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05, random_state=42, verbose=0,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1,
        )),
    ]
    meta_learner = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    return StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )


def train(X_train, y_train) -> StackingClassifier:
    logger.info("Treinando Stacking Ensemble (estudo)...")
    logger.info("  Base learners: XGBoost, LightGBM, CatBoost, RandomForest")
    logger.info("  Meta-learner:  Logistic Regression")
    logger.info(f"  Amostras de treino: {len(X_train):,}")
    model = build_stacking_model()
    model.fit(X_train, y_train)
    logger.info("Treinamento (estudo) concluido.")
    return model


# ── Persistencia ──────────────────────────────────────────────────────

def save_serving(model, scaler,
                 path_model: str = "models/serving_model.pkl",
                 path_scaler: str = "models/scaler.pkl"):
    os.makedirs(os.path.dirname(path_model), exist_ok=True)
    joblib.dump(model, path_model)
    joblib.dump(scaler, path_scaler)
    logger.info(f"Modelo de serving salvo: {path_model}")
    logger.info(f"Scaler salvo: {path_scaler}")


def save_model(model, path: str = "models/stacking_model.pkl"):
    """Salva o modelo de estudo (grande; nao usado pela API)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Modelo de estudo salvo: {path}")


def load_serving(path_model: str = "models/serving_model.pkl",
                 path_scaler: str = "models/scaler.pkl"):
    return joblib.load(path_model), joblib.load(path_scaler)
