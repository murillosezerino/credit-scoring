"""
preprocess.py — Limpeza, feature engineering, escala e (opcional) SMOTE.
Dataset: Give Me Some Credit (Kaggle).

Ordem que importa para a correcao:
1. split estratificado
2. ajustar o scaler SO no treino, transformar treino e teste
3. aplicar SMOTE sobre o treino JA ESCALADO (assim o KNN do SMOTE opera em
   distancias comparaveis e o scaler nunca e influenciado por dados sinteticos)
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

TARGET = "SeriousDlqin2yrs"

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

ENGINEERED = ["TotalLatePayments", "IncomePerDependent", "CreditUtilizationRisk"]


def feature_columns() -> list:
    """As 13 colunas de entrada do modelo, na ordem usada em treino e na API."""
    return FEATURES + ENGINEERED


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Carregando dataset: {path}")
    df = pd.read_csv(path, index_col=0)
    logger.info(f"Shape original: {df.shape}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Limpando dados...")

    # Remove duplicatas
    before = len(df)
    df = df.drop_duplicates()
    logger.info(f"Duplicatas removidas: {before - len(df)}")

    # Imputa valores ausentes
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    # Remove outliers extremos
    df = df[df["age"] >= 18]
    df = df[df["RevolvingUtilizationOfUnsecuredLines"] <= 10]
    df = df[df["DebtRatio"] <= 10]

    logger.info(f"Shape apos limpeza: {df.shape}")
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Construindo features...")

    df["TotalLatePayments"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["CreditUtilizationRisk"] = df["RevolvingUtilizationOfUnsecuredLines"] * df["DebtRatio"]

    return df


def split_scale(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split estratificado + scaler ajustado apenas no treino (sem SMOTE)."""
    cols = feature_columns()
    X = df[cols]
    y = df[TARGET]

    logger.info(f"Distribuicao original — 0: {(y == 0).sum():,} | 1: {(y == 1).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train), columns=cols, index=X_train.index
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test), columns=cols, index=X_test.index
    )

    return X_train_s, X_test_s, y_train, y_test, scaler


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    """SMOTE aplicado sobre o treino ja escalado. Usado no modelo de estudo."""
    logger.info("Aplicando SMOTE no conjunto de treino (ja escalado)...")
    smote = SMOTE(random_state=random_state)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    logger.info(f"Apos SMOTE — 0: {(y_bal == 0).sum():,} | 1: {(y_bal == 1).sum():,}")
    return X_bal, y_bal


def run(path: str):
    """Caminho do modelo de estudo: treino escalado e balanceado com SMOTE."""
    df = build_features(clean(load_data(path)))
    X_train_s, X_test_s, y_train, y_test, scaler = split_scale(df)
    X_bal, y_bal = apply_smote(X_train_s, y_train)
    return X_bal, X_test_s, y_bal, y_test, scaler
