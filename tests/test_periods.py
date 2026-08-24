import pytest

from src.periods import parse_period_code, period_label, period_year


class TestParsePeriodCode:
    def test_quarterly_code(self):
        period = parse_period_code(2025001012)
        assert period.year == 2025
        assert period.start_month == 10
        assert period.end_month == 12

    def test_first_quarter(self):
        period = parse_period_code(2002000103)
        assert period.year == 2002
        assert period.start_month == 1
        assert period.end_month == 3

    def test_monthly_code(self):
        period = parse_period_code(2000000101)
        assert period.year == 2000
        assert period.start_month == 1
        assert period.end_month == 1

    def test_invalid_month_raises(self):
        with pytest.raises(ValueError):
            parse_period_code(2000001395)

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError):
            parse_period_code(123)


class TestPeriodHelpers:
    def test_period_year(self):
        assert period_year(2025000406) == 2025

    def test_period_label_quarterly(self):
        assert period_label(2025001012) == "2025年10-12月期"

    def test_period_label_monthly(self):
        assert period_label(2000000101) == "2000年1月"
