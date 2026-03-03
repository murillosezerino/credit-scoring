"""
main.py — Orquestrador do pipeline de Credit Scoring.
Executa: preprocessamento -> treino -> avaliacao -> salva modelo.
"""

import logging
from src.preprocess import run as preprocess
from src.train import train, save_model
from src.evaluate import evaluate, plot_roc_curve, explain_shap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(module)s — %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = "data/cs-training.csv"


def main():
    logger.info("=" * 55)
    logger.info("Credit Scoring — Stacking Ensemble Pipeline")
    logger.info("=" * 55)

    # 1. Preprocessamento
    logger.info("[1/3] Preprocessando dados...")
    X_train, X_test, y_train, y_test, scaler = preprocess(DATA_PATH)

    # 2. Treino
    logger.info("[2/3] Treinando modelo...")
    model = train(X_train, y_train)
    save_model(model, scaler)

    # 3. Avaliacao
    logger.info("[3/3] Avaliando modelo...")
    metrics = evaluate(model, X_test, y_test)
    plot_roc_curve(model, X_test, y_test)
    explain_shap(model, X_test)

    logger.info("=" * 55)
    logger.info(f"AUC-ROC final: {metrics['auc_roc']:.4f}")
    logger.info(f"AUC-PR  final: {metrics['auc_pr']:.4f}")
    logger.info("Pipeline concluido.")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()