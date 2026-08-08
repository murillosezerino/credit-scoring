"""
make_sample.py — Gera um sample SINTETICO com o schema do dataset
Give Me Some Credit, para o pipeline rodar out-of-the-box sem baixar o Kaggle.

Isto e dado FALSO, para smoke test e para o modelo do demo hospedado.
As metricas de estudo (AUC ~0.80) vem do dataset real do Kaggle, nao daqui.
"""

import os
import numpy as np
import pandas as pd

OUT = "data/sample/cs-training-sample.csv"


def main(n: int = 6000, seed: int = 42, out: str = OUT):
    rng = np.random.RandomState(seed)

    default = rng.binomial(1, 0.07, n)  # ~7% de positivos, como no dataset real

    df = pd.DataFrame({
        "SeriousDlqin2yrs": default,
        "RevolvingUtilizationOfUnsecuredLines": np.clip(
            rng.beta(2, 5, n) + default * 0.20 * rng.rand(n), 0, 1.5),
        "age": rng.randint(21, 75, n),
        "NumberOfTime30-59DaysPastDueNotWorse": rng.poisson(0.2 + default * 0.8, n),
        "DebtRatio": np.clip(rng.gamma(2.0, 0.15, n) + default * 0.2, 0, 5),
        "MonthlyIncome": np.where(
            rng.rand(n) > 0.10, rng.lognormal(8.6, 0.5, n), np.nan),
        "NumberOfOpenCreditLinesAndLoans": rng.poisson(8, n),
        "NumberOfTimes90DaysLate": rng.poisson(0.05 + default * 0.6, n),
        "NumberRealEstateLoansOrLines": rng.poisson(1.0, n),
        "NumberOfTime60-89DaysPastDueNotWorse": rng.poisson(0.05 + default * 0.5, n),
        "NumberOfDependents": np.where(
            rng.rand(n) > 0.05, rng.poisson(0.8, n).astype(float), np.nan),
    })

    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out)  # o indice vira a coluna id (load usa index_col=0)
    print(f"Wrote {out}: {df.shape}, positives={int(default.sum())}")


if __name__ == "__main__":
    main()
