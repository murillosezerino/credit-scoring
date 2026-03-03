"""
api.py — API REST com FastAPI para servir predicoes do modelo de credit scoring.
"""

import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Scoring API",
    description="API para predicao de risco de inadimplencia usando Stacking Ensemble",
    version="1.0.0",
)

# Carrega modelo e scaler na inicializacao
try:
    model  = joblib.load("models/stacking_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    logger.info("Modelo carregado com sucesso.")
except Exception as e:
    model  = None
    scaler = None
    logger.warning(f"Modelo nao encontrado: {e}")


class CreditRequest(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, description="Utilizacao de credito rotativo")
    age: int                                     = Field(..., ge=18, le=100, description="Idade do cliente")
    NumberOfTime30_59DaysPastDueNotWorse: int    = Field(..., ge=0, description="Atrasos 30-59 dias")
    DebtRatio: float                             = Field(..., ge=0, description="Razao de divida")
    MonthlyIncome: float                         = Field(..., ge=0, description="Renda mensal")
    NumberOfOpenCreditLinesAndLoans: int         = Field(..., ge=0, description="Linhas de credito abertas")
    NumberOfTimes90DaysLate: int                 = Field(..., ge=0, description="Atrasos acima de 90 dias")
    NumberRealEstateLoansOrLines: int            = Field(..., ge=0, description="Emprestimos imobiliarios")
    NumberOfTime60_89DaysPastDueNotWorse: int    = Field(..., ge=0, description="Atrasos 60-89 dias")
    NumberOfDependents: int                      = Field(..., ge=0, description="Numero de dependentes")


class CreditResponse(BaseModel):
    default_probability: float
    risk_score: int
    risk_label: str
    recommendation: str


def calculate_risk(proba: float) -> tuple:
    score = int((1 - proba) * 1000)
    if proba < 0.10:
        return score, "Baixo Risco",    "Aprovado"
    elif proba < 0.30:
        return score, "Medio Risco",   "Aprovado com restricoes"
    elif proba < 0.50:
        return score, "Alto Risco",    "Analise manual recomendada"
    else:
        return score, "Critico",       "Reprovado"


@app.get("/")
def root():
    return {"status": "ok", "model": "Stacking Ensemble Credit Scoring v1.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo nao carregado. Treine primeiro.")

    data = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines":  request.RevolvingUtilizationOfUnsecuredLines,
        "age":                                   request.age,
        "NumberOfTime30-59DaysPastDueNotWorse":  request.NumberOfTime30_59DaysPastDueNotWorse,
        "DebtRatio":                             request.DebtRatio,
        "MonthlyIncome":                         request.MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans":        request.NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate":               request.NumberOfTimes90DaysLate,
        "NumberRealEstateLoansOrLines":          request.NumberRealEstateLoansOrLines,
        "NumberOfTime60-89DaysPastDueNotWorse":  request.NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfDependents":                    request.NumberOfDependents,
    }])

    # Engenharia de features
    data["TotalLatePayments"]     = (
        data["NumberOfTime30-59DaysPastDueNotWorse"]
        + data["NumberOfTime60-89DaysPastDueNotWorse"]
        + data["NumberOfTimes90DaysLate"]
    )
    data["IncomePerDependent"]    = data["MonthlyIncome"] / (data["NumberOfDependents"] + 1)
    data["CreditUtilizationRisk"] = data["RevolvingUtilizationOfUnsecuredLines"] * data["DebtRatio"]

    data_scaled = pd.DataFrame(
        scaler.transform(data), columns=data.columns
    )

    proba = model.predict_proba(data_scaled)[0][1]
    score, label, recommendation = calculate_risk(proba)

    return CreditResponse(
        default_probability=round(float(proba), 4),
        risk_score=score,
        risk_label=label,
        recommendation=recommendation,
    )