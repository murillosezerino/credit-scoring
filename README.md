# Credit Scoring — Stacking Ensemble

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-green)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow)
![SHAP](https://img.shields.io/badge/SHAP-0.44-purple)

Modelo de **credit scoring** para previsao de inadimplencia utilizando **Stacking Ensemble** com interpretabilidade via **SHAP**. AUC-ROC de **~0.80** em 150.000+ registros.

## Arquitetura do Pipeline

```
Dataset (Kaggle: Give Me Some Credit)
    │
    ▼
[1] Preprocessamento (preprocess.py)
    ├── Limpeza (duplicatas, outliers, imputacao)
    ├── Feature Engineering (3 features derivadas)
    └── Balanceamento (SMOTE no treino)
    │
    ▼
[2] Treinamento (train.py)
    ├── Base Learners:
    │   ├── XGBoost (300 estimators, depth 6)
    │   ├── LightGBM (300 estimators, depth 6)
    │   ├── CatBoost (300 iterations, depth 6)
    │   └── Random Forest (200 estimators, depth 8)
    └── Meta-Learner: Logistic Regression (CV=5)
    │
    ▼
[3] Avaliacao (evaluate.py)
    ├── Metricas: AUC-ROC, AUC-PR, Classification Report
    ├── ROC Curve (salva em models/roc_curve.png)
    └── SHAP Summary (salva em models/shap_summary.png)
```

## Features

### Features Originais (dataset)

| Feature | Descricao |
|---|---|
| RevolvingUtilizationOfUnsecuredLines | Utilizacao de credito rotativo |
| age | Idade do tomador |
| NumberOfTime30-59DaysPastDueNotWorse | Atrasos de 30-59 dias |
| DebtRatio | Razao divida/renda |
| MonthlyIncome | Renda mensal |
| NumberOfOpenCreditLinesAndLoans | Linhas de credito abertas |
| NumberOfTimes90DaysLate | Atrasos de 90+ dias |
| NumberRealEstateLoansOrLines | Emprestimos imobiliarios |
| NumberOfTime60-89DaysPastDueNotWorse | Atrasos de 60-89 dias |
| NumberOfDependents | Numero de dependentes |

### Features Engenheiradas

| Feature | Formula |
|---|---|
| TotalLatePayments | Soma de todos os atrasos (30-59 + 60-89 + 90+) |
| IncomePerDependent | Renda / (Dependentes + 1) |
| CreditUtilizationRisk | Utilizacao rotativo x DebtRatio |

## Estrutura do Projeto

```
├── main.py              # Orquestrador do pipeline
├── src/
│   ├── preprocess.py    # Limpeza, features e SMOTE
│   ├── train.py         # Stacking Ensemble
│   ├── evaluate.py      # Metricas e SHAP
│   └── api.py           # API REST (FastAPI)
├── tests/               # Testes unitarios
├── models/              # Modelo e artefatos salvos
├── data/                # Dataset (nao versionado)
├── requirements.txt
└── .github/workflows/
    └── ci.yml           # CI/CD automatizado
```

## Como Rodar

### Pre-requisitos

- Python 3.11+

### Setup

```bash
# Clonar o repositorio
git clone https://github.com/murillosezerino/credit-scoring.git
cd credit-scoring

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Dataset

Baixe o dataset **"Give Me Some Credit"** do Kaggle e coloque em `data/cs-training.csv`:

```
https://www.kaggle.com/c/GiveMeSomeCredit/data
```

### Executar o Pipeline

```bash
# Rodar pipeline completo (preprocess -> train -> evaluate)
python main.py
```

### API REST

```bash
# Servir modelo via FastAPI
uvicorn src.api:app --reload
```

## Resultados

| Metrica | Valor |
|---|---|
| AUC-ROC | ~0.80 |
| AUC-PR | Varia por threshold |
| Modelo | Stacking Ensemble (4 base + meta) |
| Interpretabilidade | SHAP TreeExplainer |

## Licenca

MIT
