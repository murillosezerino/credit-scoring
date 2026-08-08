import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import src.api as api_module
from src.api import app, calculate_risk
from src.preprocess import feature_columns

VALID_PAYLOAD = {
    "RevolvingUtilizationOfUnsecuredLines": 0.45,
    "age": 35,
    "NumberOfTime30_59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.32,
    "MonthlyIncome": 5000,
    "NumberOfOpenCreditLinesAndLoans": 6,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60_89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1,
}


def _fit_stub():
    """Modelo e scaler minimos, com as 13 colunas na ordem que a API monta."""
    cols = feature_columns()
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.rand(200, len(cols)), columns=cols)
    y = rng.randint(0, 2, 200)
    scaler = StandardScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=cols)
    model = LogisticRegression(max_iter=200).fit(X_scaled, y)
    return model, scaler


class TestCalculateRisk:
    def test_low_risk(self):
        score, label, rec = calculate_risk(0.05)
        assert label == "Baixo Risco"
        assert rec == "Aprovado"
        assert score > 0

    def test_medium_risk(self):
        score, label, rec = calculate_risk(0.20)
        assert label == "Medio Risco"
        assert rec == "Aprovado com restricoes"

    def test_high_risk(self):
        score, label, rec = calculate_risk(0.40)
        assert label == "Alto Risco"
        assert rec == "Analise manual recomendada"

    def test_critical_risk(self):
        score, label, rec = calculate_risk(0.70)
        assert label == "Critico"
        assert rec == "Reprovado"

    def test_score_decreases_with_higher_proba(self):
        score_low, _, _ = calculate_risk(0.05)
        score_high, _, _ = calculate_risk(0.90)
        assert score_low > score_high

    def test_boundary_010(self):
        _, label, _ = calculate_risk(0.10)
        assert label == "Medio Risco"

    def test_boundary_030(self):
        _, label, _ = calculate_risk(0.30)
        assert label == "Alto Risco"

    def test_boundary_050(self):
        _, label, _ = calculate_risk(0.50)
        assert label == "Critico"


class TestEndpoints:
    def test_root(self):
        r = TestClient(app).get("/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health(self):
        r = TestClient(app).get("/health")
        assert r.status_code == 200
        assert "model_loaded" in r.json()

    def test_predict_ok(self, monkeypatch):
        model, scaler = _fit_stub()
        monkeypatch.setattr(api_module, "model", model)
        monkeypatch.setattr(api_module, "scaler", scaler)
        r = TestClient(app).post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"default_probability", "risk_score", "risk_label", "recommendation"}
        assert 0.0 <= body["default_probability"] <= 1.0

    def test_predict_validation_422(self):
        bad = dict(VALID_PAYLOAD)
        bad["age"] = 10  # viola ge=18
        r = TestClient(app).post("/predict", json=bad)
        assert r.status_code == 422

    def test_predict_503_when_model_missing(self, monkeypatch):
        monkeypatch.setattr(api_module, "model", None)
        r = TestClient(app).post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 503
