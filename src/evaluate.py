"""
evaluate.py — Metricas de avaliacao e interpretabilidade com SHAP.
"""

import logging
import numpy as np
import pandas as pd
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
    auc_pr  = average_precision_score(y_test, y_proba)

    logger.info(f"  AUC-ROC:        {auc_roc:.4f}")
    logger.info(f"  AUC-PR:         {auc_pr:.4f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    metrics = {
        "auc_roc": auc_roc,
        "auc_pr":  auc_pr,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    return metrics


def plot_roc_curve(model, X_test, y_test, save_path: str = "models/roc_curve.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("ROC Curve — Stacking Ensemble")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"ROC curve salva: {save_path}")


def explain_shap(model, X_test, save_path: str = "models/shap_summary.png"):
    logger.info("Calculando SHAP values...")

    # Usa o XGBoost (primeiro base learner) para SHAP
    xgb_model = model.named_estimators_["xgb"]
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test,
        plot_type="bar",
        show=False
    )
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"SHAP summary salvo: {save_path}")

    return shap_values