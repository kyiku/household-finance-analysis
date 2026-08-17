from pathlib import Path

import pytest

from src.analysis import (
    FINE_AGE_ORDER,
    coarse_age_surplus_rate,
    latest_breakdown_share_by_age,
    latest_surplus_stats_by_quintile,
    yearly_mean_pivot,
)
from src.benchmark import latest_mean_by
from src.data_loader import (
    load_income_expense_by_age,
    load_savings_breakdown_by_age,
    load_savings_debt_by_age,
    load_surplus_by_quintile,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

QUINTILE_ORDER = ["年収五分位1", "年収五分位2", "年収五分位3", "年収五分位4", "年収五分位5"]


class TestLatestSurplusStatsByQuintile:
    def test_returns_all_quintiles_with_stats(self):
        df = load_surplus_by_quintile(DATA_DIR)
        stats = latest_surplus_stats_by_quintile(df)
        assert list(stats.index) == QUINTILE_ORDER
        assert "黒字率" in stats.columns
        assert "平均消費性向" in stats.columns

    def test_higher_income_saves_more(self):
        # データから期待される傾向: 五分位5の黒字率 > 五分位1
        df = load_surplus_by_quintile(DATA_DIR)
        stats = latest_surplus_stats_by_quintile(df)
        assert stats.loc["年収五分位5", "黒字率"] > stats.loc["年収五分位1", "黒字率"]

    def test_no_nan_in_latest_stats(self):
        # 回帰テスト: 年途中(部分年)データが混ざってもNaNを出さない
        df = load_surplus_by_quintile(DATA_DIR)
        stats = latest_surplus_stats_by_quintile(df)
        assert not stats.isna().any().any()


class TestYearlyMeanPivot:
    def test_pivot_shape(self):
        df = load_surplus_by_quintile(DATA_DIR)
        pivot = yearly_mean_pivot(df, group_col="年間収入五分位", value_col="黒字率")
        assert list(pivot.columns) == QUINTILE_ORDER
        assert pivot.index.name == "年"

    def test_works_for_age_data(self):
        df = load_savings_debt_by_age(DATA_DIR)
        pivot = yearly_mean_pivot(df, group_col="年齢階級", value_col="貯蓄")
        assert "40～49歳" in pivot.columns


class TestLatestBreakdownShareByAge:
    def test_shares_sum_to_100(self):
        df = load_savings_breakdown_by_age(DATA_DIR)
        share = latest_breakdown_share_by_age(df)
        sums = share.sum(axis=1)
        assert all(abs(s - 100.0) < 1e-6 for s in sums)

    def test_age_order(self):
        df = load_savings_breakdown_by_age(DATA_DIR)
        share = latest_breakdown_share_by_age(df)
        assert list(share.index)[0] == "29歳以下"


class TestFineAgeBenchmark:
    def test_all_fine_brackets_have_values(self):
        # 回帰テスト: 「特徴3」グラフの若年層バーがNaNで欠落しないこと
        df = load_income_expense_by_age(DATA_DIR)
        bench = latest_mean_by(df, "年齢階級", ["黒字率"]).reindex(FINE_AGE_ORDER)
        assert not bench["黒字率"].isna().any()


class TestCoarseAgeSurplusRate:
    def test_maps_coarse_bracket(self):
        df = load_income_expense_by_age(DATA_DIR)
        rate = coarse_age_surplus_rate(df, "40～49歳")
        assert rate is not None
        assert -100.0 < rate < 100.0

    def test_young_bracket_available_despite_partial_year(self):
        # 回帰テスト: 最新暦年(2026年)に若年層のデータがまだ無くても、
        # 直近1年分の公表済みデータで比較値を返せること
        df = load_income_expense_by_age(DATA_DIR)
        rate = coarse_age_surplus_rate(df, "29歳以下")
        assert rate is not None
        assert -100.0 < rate < 100.0

    def test_unavailable_bracket_returns_none(self):
        df = load_income_expense_by_age(DATA_DIR)
        assert coarse_age_surplus_rate(df, "70歳以上") is None

    def test_unknown_bracket_raises(self):
        df = load_income_expense_by_age(DATA_DIR)
        with pytest.raises(ValueError):
            coarse_age_surplus_rate(df, "存在しない年代")
