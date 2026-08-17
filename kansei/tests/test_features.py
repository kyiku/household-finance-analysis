import pytest

from src.features import compute_ratios, income_class_of, surplus_rate


class TestIncomeClassOf:
    def test_below_lowest_bound(self):
        assert income_class_of(150) == "200万円未満"

    def test_middle_class(self):
        assert income_class_of(620) == "600～650万円"

    def test_boundary_is_inclusive_lower(self):
        assert income_class_of(600) == "600～650万円"

    def test_wide_upper_class(self):
        assert income_class_of(1100) == "1000～1250万円"

    def test_above_highest_bound(self):
        assert income_class_of(2000) == "1500万円以上"

    def test_zero_income_raises(self):
        with pytest.raises(ValueError):
            income_class_of(0)

    def test_negative_income_raises(self):
        with pytest.raises(ValueError):
            income_class_of(-10)


class TestComputeRatios:
    def test_basic_ratios(self):
        ratios = compute_ratios(annual_income_man=600, savings_man=1200, debt_man=300)
        assert ratios["貯蓄/年収倍率"] == pytest.approx(2.0)
        assert ratios["負債/年収倍率"] == pytest.approx(0.5)

    def test_zero_income_raises(self):
        with pytest.raises(ValueError):
            compute_ratios(annual_income_man=0, savings_man=100, debt_man=0)

    def test_negative_savings_raises(self):
        with pytest.raises(ValueError):
            compute_ratios(annual_income_man=500, savings_man=-1, debt_man=0)


class TestSurplusRate:
    def test_positive_surplus(self):
        assert surplus_rate(monthly_disposable_man=30, monthly_expense_man=24) == pytest.approx(20.0)

    def test_negative_surplus_allowed(self):
        assert surplus_rate(monthly_disposable_man=30, monthly_expense_man=33) == pytest.approx(-10.0)

    def test_zero_disposable_raises(self):
        with pytest.raises(ValueError):
            surplus_rate(monthly_disposable_man=0, monthly_expense_man=10)
