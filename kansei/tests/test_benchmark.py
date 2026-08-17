import pandas as pd
import pytest

from src.benchmark import (
    annual_period_count,
    build_comparison_table,
    latest_mean_by,
    latest_year,
)


@pytest.fixture
def sample_df():
    """A は2025年の4四半期のみ、B は2026年Q1まで公表済み(公表時期のずれを再現)。"""
    quarters_2025 = [2025000103, 2025000406, 2025000709, 2025001012]
    rows = [
        *[
            {"年": 2025, "時期": code, "階級": "A", "貯蓄": value, "負債": 50}
            for code, value in zip(quarters_2025, [100, 110, 120, 130])
        ],
        *[
            {"年": 2025, "時期": code, "階級": "B", "貯蓄": value, "負債": 60}
            for code, value in zip(quarters_2025, [200, 210, 220, 230])
        ],
        {"年": 2026, "時期": 2026000103, "階級": "B", "貯蓄": 240, "負債": 60},
    ]
    return pd.DataFrame(rows)


class TestLatestYear:
    def test_returns_max_year(self, sample_df):
        assert latest_year(sample_df) == 2026


class TestAnnualPeriodCount:
    def test_quarterly_data(self, sample_df):
        assert annual_period_count(sample_df) == 4

    def test_monthly_data(self):
        df = pd.DataFrame({"時期": [2025000101, 2025000202]})
        assert annual_period_count(df) == 12


class TestLatestMeanBy:
    def test_uses_each_groups_latest_window(self, sample_df):
        """グループごとに「そのグループで公表済みの直近4四半期」を使う。

        暦年でフィルタすると2026年Q1しかないBの値が偏り、Aは全欠損(NaN)になる。
        """
        result = latest_mean_by(sample_df, group_col="階級", value_cols=["貯蓄", "負債"])
        assert result.loc["A", "貯蓄"] == pytest.approx(115.0)  # 2025年4四半期の平均
        assert result.loc["B", "貯蓄"] == pytest.approx(225.0)  # 2025Q2〜2026Q1の平均
        assert not result.isna().any().any()

    def test_explicit_n_periods(self, sample_df):
        result = latest_mean_by(sample_df, group_col="階級", value_cols=["貯蓄"], n_periods=1)
        assert result.loc["A", "貯蓄"] == pytest.approx(130.0)
        assert result.loc["B", "貯蓄"] == pytest.approx(240.0)

    def test_does_not_mutate_input(self, sample_df):
        before = sample_df.copy()
        latest_mean_by(sample_df, group_col="階級", value_cols=["貯蓄"])
        pd.testing.assert_frame_equal(sample_df, before)


class TestBuildComparisonTable:
    def test_table_shape_and_values(self):
        user = {"貯蓄(万円)": 500.0, "負債(万円)": 100.0}
        profile = {"貯蓄(万円)": 800.0, "負債(万円)": 300.0}
        table = build_comparison_table(user, profile, benchmark_label="同年代平均")
        assert list(table.columns) == ["あなた", "同年代平均"]
        assert table.loc["貯蓄(万円)", "あなた"] == pytest.approx(500.0)
        assert table.loc["負債(万円)", "同年代平均"] == pytest.approx(300.0)
