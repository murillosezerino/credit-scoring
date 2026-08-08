"""
api.py — API REST com FastAPI para servir predicoes do modelo de credit scoring.
Carrega o modelo de serving (LightGBM calibrado) e o scaler.
"""

import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Scoring API",
    description="Predicao de risco de inadimplencia (LightGBM calibrado), servida com FastAPI.",
    version="1.1.0",
)

# Carrega modelo de serving e scaler na inicializacao
try:
    model = joblib.load("models/serving_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    logger.info("Modelo de serving carregado com sucesso.")
except Exception as e:
    model = None
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
        return score, "Baixo Risco",  "Aprovado"
    elif proba < 0.30:
        return score, "Medio Risco",  "Aprovado com restricoes"
    elif proba < 0.50:
        return score, "Alto Risco",   "Analise manual recomendada"
    else:
        return score, "Critico",      "Reprovado"


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "Credit Scoring API v1.1 (calibrated LightGBM)",
        "note": "Demo model trained on a synthetic sample; study metrics come from the full Kaggle dataset.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo nao carregado. Treine primeiro (python main.py).")

    data = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines":  request.RevolvingUtilizationOfUnsecuredLines,
        "age":                                   request.age,
        "NumberOfTime30-59DaysPastDueNotWorse":  request.NumberOfTime30_59DaysPastDueNotWorse,
        "DebtRatio":                             request.DebtRatio,
        "MonthlyIncome":                         request.MonthlyIncome,
        "NumberOfOpenCreditLinesAndLoans":       request.NumberOfOpenCreditLinesAndLoans,
        "NumberOfTimes90DaysLate":               request.NumberOfTimes90DaysLate,
        "NumberRealEstateLoansOrLines":          request.NumberRealEstateLoansOrLines,
        "NumberOfTime60-89DaysPastDueNotWorse":  request.NumberOfTime60_89DaysPastDueNotWorse,
        "NumberOfDependents":                    request.NumberOfDependents,
    }])

    # Engenharia de features (mesma logica do treino)
    data["TotalLatePayments"] = (
        data["NumberOfTime30-59DaysPastDueNotWorse"]
        + data["NumberOfTime60-89DaysPastDueNotWorse"]
        + data["NumberOfTimes90DaysLate"]
    )
    data["IncomePerDependent"] = data["MonthlyIncome"] / (data["NumberOfDependents"] + 1)
    data["CreditUtilizationRisk"] = data["RevolvingUtilizationOfUnsecuredLines"] * data["DebtRatio"]

    data_scaled = pd.DataFrame(scaler.transform(data), columns=data.columns)

    proba = float(model.predict_proba(data_scaled)[0][1])
    score, label, recommendation = calculate_risk(proba)

    return CreditResponse(
        default_probability=round(proba, 4),
        risk_score=score,
        risk_label=label,
        recommendation=recommendation,
    )
