from src.api import calculate_risk


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
