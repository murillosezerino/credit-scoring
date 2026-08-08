"""
evaluate.py — Metricas de avaliacao e interpretabilidade com SHAP.
Backend Agg para rodar headless (CI, servidores sem display).
"""

import logging
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    RocCurveDisplay,
)

logger = logging.getLogger(__name__)


def evaluate(model, X_test, y_test) -> dict:
    logger.info("Avaliando modelo...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)

    logger.info(f"  AUC-ROC:        {auc_roc:.4f}")
    logger.info(f"  AUC-PR:         {auc_pr:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def plot_roc_curve(model, X_test, y_test,
                   save_path: str = "models/roc_curve.png",
                   title: str = "ROC Curve"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"ROC curve salva: {save_path}")


def explain_shap(tree_model, X_test, save_path: str = "models/shap_summary.png"):
    """SHAP a partir de um modelo de arvore (LGBM servido ou base XGB do estudo)."""
    logger.info("Calculando SHAP values...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_test)
    # Classificacao binaria: algumas versoes retornam lista [classe0, classe1]
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"SHAP summary salvo: {save_path}")

    return shap_values
