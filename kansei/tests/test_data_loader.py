from pathlib import Path

import pytest

from src.data_loader import (
    load_income_expense_by_age,
    load_savings_breakdown_by_age,
    load_savings_debt_by_age,
    load_savings_debt_by_income,
    load_surplus_by_quintile,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestLoadSavingsDebtByIncome:
    def test_wide_columns_and_year(self):
        df = load_savings_debt_by_income(DATA_DIR)
        for col in ["年間収入階級", "時期", "年", "年間収入", "貯蓄", "負債", "貯蓄/年収倍率", "負債/年収倍率"]:
            assert col in df.columns
        assert df["年"].min() == 2002
        assert df["年"].max() >= 2025

    def test_ratios_are_consistent(self):
        df = load_savings_debt_by_income(DATA_DIR)
        row = df.iloc[0]
        assert row["貯蓄/年収倍率"] == pytest.approx(row["貯蓄"] / row["年間収入"])


class TestLoadSavingsDebtByAge:
    def test_wide_columns(self):
        df = load_savings_debt_by_age(DATA_DIR)
        for col in ["年齢階級", "時期", "年", "年間収入", "貯蓄", "負債"]:
            assert col in df.columns
        assert set(df["年齢階級"]) == {
            "29歳以下", "30～39歳", "40～49歳", "50～59歳", "60～69歳", "70歳以上",
        }


class TestLoadIncomeExpenseByAge:
    def test_contains_surplus_rate(self):
        df = load_income_expense_by_age(DATA_DIR)
        assert "黒字率" in df.columns
        assert "年" in df.columns


class TestLoadSavingsBreakdownByAge:
    def test_breakdown_columns(self):
        df = load_savings_breakdown_by_age(DATA_DIR)
        for col in ["通貨性預貯金", "定期性預貯金", "生命保険など", "有価証券", "金融機関外"]:
            assert col in df.columns


class TestLoadSurplusByQuintile:
    def test_quintile_column(self):
        df = load_surplus_by_quintile(DATA_DIR)
        assert "年間収入五分位" in df.columns
        assert "黒字率" in df.columns
        assert df["年間収入五分位"].nunique() == 5


class TestMissingFile:
    def test_missing_dir_raises_friendly_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_savings_debt_by_income(tmp_path)


class TestDataQuality:
    def test_columns_axis_name_is_cleared(self):
        # pivot_table 由来の columns.name="項目" が表示に紛れ込まないこと
        df = load_savings_debt_by_income(DATA_DIR)
        assert df.columns.name is None

    def test_duplicate_rows_raise(self, tmp_path):
        csv = tmp_path / "kakei_savings_debt_by_income.csv"
        csv.write_text(
            "項目,年間収入階級,時期,値,単位\n"
            "年間収入,200万円未満,2002000103,150,万円\n"
            "年間収入,200万円未満,2002000103,160,万円\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_savings_debt_by_income(tmp_path)
