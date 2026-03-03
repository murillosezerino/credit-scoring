"""
preprocess.py — Limpeza, Feature Engineering e balanceamento com SMOTE.
Dataset: Give Me Some Credit (Kaggle)
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


def split_and_balance(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    logger.info("Separando features e target...")

    feature_cols = FEATURES + ["TotalLatePayments", "IncomePerDependent", "CreditUtilizationRisk"]
    X = df[feature_cols]
    y = df[TARGET]

    logger.info(f"Distribuicao original — 0: {(y==0).sum():,} | 1: {(y==1).sum():,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # SMOTE apenas no treino
    logger.info("Aplicando SMOTE no conjunto de treino...")
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    logger.info(f"Apos SMOTE — 0: {(y_train_bal==0).sum():,} | 1: {(y_train_bal==1).sum():,}")

    # Normalizar
    scaler = StandardScaler()
    X_train_bal = pd.DataFrame(
        scaler.fit_transform(X_train_bal), columns=X_train_bal.columns
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    return X_train_bal, X_test, y_train_bal, y_test, scaler


def run(path: str):
    df = load_data(path)
    df = clean(df)
    df = build_features(df)
    return split_and_balance(df)