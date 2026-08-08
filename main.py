"""
main.py — Orquestrador do pipeline de Credit Scoring.

Roda out-of-the-box num sample sintetico (data/sample/cs-training-sample.csv).
Para o dataset real do Kaggle:
    python scripts/download_data.py
    DATA_PATH=data/cs-training.csv python main.py

Etapas: preprocess -> serving model (calibrado, salvo para a API) -> estudo (stacking).
Defina TRAIN_STUDY=0 para pular o stacking e rodar so o serving (mais rapido).
"""

import logging
import os

from src.preprocess import load_data, clean, build_features, split_scale, apply_smote
from src.train import train_serving, train_explainer, save_serving, train, save_model
from src.evaluate import evaluate, plot_roc_curve, explain_shap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(module)s — %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = os.getenv("DATA_PATH", "data/sample/cs-training-sample.csv")
TRAIN_STUDY = os.getenv("TRAIN_STUDY", "1") == "1"


def main():
    logger.info("=" * 55)
    logger.info("Credit Scoring Pipeline")
    logger.info(f"Dataset: {DATA_PATH}")
    logger.info("=" * 55)

    # 1. Preprocessamento
    logger.info("[1/3] Preprocessando dados...")
    df = build_features(clean(load_data(DATA_PATH)))
    X_train, X_test, y_train, y_test, scaler = split_scale(df)

    # 2. Modelo de serving (o que a API usa) + artefatos de interpretabilidade
    logger.info("[2/3] Treinando modelo de serving (calibrado)...")
    serving = train_serving(X_train, y_train)
    save_serving(serving, scaler)
    m_serving = evaluate(serving, X_test, y_test)
    plot_roc_curve(serving, X_test, y_test, "models/roc_serving.png",
                   title="ROC — Serving (calibrated LGBM)")
    explainer = train_explainer(X_train, y_train)
    explain_shap(explainer, X_test, "models/shap_serving.png")

    # 3. Modelo de estudo (stacking ensemble)
    if TRAIN_STUDY:
        logger.info("[3/3] Treinando modelo de estudo (stacking)...")
        X_bal, y_bal = apply_smote(X_train, y_train)
        stacking = train(X_bal, y_bal)
        save_model(stacking)
        m_study = evaluate(stacking, X_test, y_test)
        plot_roc_curve(stacking, X_test, y_test, "models/roc_stacking.png",
                       title="ROC — Stacking Ensemble")
        explain_shap(stacking.named_estimators_["xgb"], X_test, "models/shap_stacking.png")
    else:
        logger.info("[3/3] TRAIN_STUDY=0 — pulando o stacking.")
        m_study = None

    logger.info("=" * 55)
    logger.info(f"Serving  AUC-ROC: {m_serving['auc_roc']:.4f} | AUC-PR: {m_serving['auc_pr']:.4f}")
    if m_study:
        logger.info(f"Study    AUC-ROC: {m_study['auc_roc']:.4f} | AUC-PR: {m_study['auc_pr']:.4f}")
    logger.info("Pipeline concluido.")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()
