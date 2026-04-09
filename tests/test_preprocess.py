import pandas as pd
import numpy as np
from src.preprocess import clean, build_features, FEATURES, TARGET


def _make_sample_df(n=100):
    """Create a minimal DataFrame that mimics the Give Me Some Credit dataset."""
    np.random.seed(42)
    return pd.DataFrame({
        TARGET: np.random.choice([0, 1], n, p=[0.93, 0.07]),
        "RevolvingUtilizationOfUnsecuredLines": np.random.uniform(0, 1, n),
        "age": np.random.randint(21, 70, n),
        "NumberOfTime30-59DaysPastDueNotWorse": np.random.randint(0, 5, n),
        "DebtRatio": np.random.uniform(0, 2, n),
        "MonthlyIncome": np.where(
            np.random.random(n) > 0.1, np.random.uniform(2000, 15000, n), np.nan
        ),
        "NumberOfOpenCreditLinesAndLoans": np.random.randint(0, 15, n),
        "NumberOfTimes90DaysLate": np.random.randint(0, 3, n),
        "NumberRealEstateLoansOrLines": np.random.randint(0, 5, n),
        "NumberOfTime60-89DaysPastDueNotWorse": np.random.randint(0, 3, n),
        "NumberOfDependents": np.where(
            np.random.random(n) > 0.05, np.random.randint(0, 4, n), np.nan
        ),
    })


class TestClean:
    def test_removes_underage(self):
        df = _make_sample_df()
        df.loc[0, "age"] = 16
        cleaned = clean(df)
        assert (cleaned["age"] >= 18).all()

    def test_fills_monthly_income_nulls(self):
        df = _make_sample_df()
        cleaned = clean(df)
        assert cleaned["MonthlyIncome"].isna().sum() == 0

    def test_fills_dependents_nulls(self):
        df = _make_sample_df()
        cleaned = clean(df)
        assert cleaned["NumberOfDependents"].isna().sum() == 0

    def test_removes_extreme_utilization(self):
        df = _make_sample_df()
        df.loc[0, "RevolvingUtilizationOfUnsecuredLines"] = 50
        cleaned = clean(df)
        assert (cleaned["RevolvingUtilizationOfUnsecuredLines"] <= 10).all()

    def test_removes_extreme_debt_ratio(self):
        df = _make_sample_df()
        df.loc[0, "DebtRatio"] = 100
        cleaned = clean(df)
        assert (cleaned["DebtRatio"] <= 10).all()


class TestBuildFeatures:
    def test_adds_total_late_payments(self):
        df = _make_sample_df()
        df = clean(df)
        result = build_features(df)
        assert "TotalLatePayments" in result.columns

    def test_adds_income_per_dependent(self):
        df = _make_sample_df()
        df = clean(df)
        result = build_features(df)
        assert "IncomePerDependent" in result.columns
        assert result["IncomePerDependent"].isna().sum() == 0

    def test_adds_credit_utilization_risk(self):
        df = _make_sample_df()
        df = clean(df)
        result = build_features(df)
        assert "CreditUtilizationRisk" in result.columns

    def test_total_late_is_sum_of_components(self):
        df = _make_sample_df()
        df = clean(df)
        result = build_features(df)
        expected = (
            result["NumberOfTime30-59DaysPastDueNotWorse"]
            + result["NumberOfTime60-89DaysPastDueNotWorse"]
            + result["NumberOfTimes90DaysLate"]
        )
        pd.testing.assert_series_equal(
            result["TotalLatePayments"].reset_index(drop=True),
            expected.reset_index(drop=True),
        )
