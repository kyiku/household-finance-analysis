from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import load_savings_distribution
from src.percentile import parse_class_bounds, savings_percentile, savings_value_at

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestParseClassBounds:
    def test_under_class(self):
        assert parse_class_bounds("100万円未満") == (0.0, 100.0)

    def test_range_class(self):
        assert parse_class_bounds("1000～1200万円") == (1000.0, 1200.0)

    def test_open_top_class(self):
        lower, upper = parse_class_bounds("4000万円以上")
        assert lower == 4000.0
        assert upper == float("inf")

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError):
            parse_class_bounds("その他")


@pytest.fixture
def simple_dist():
    # 4階級で25%ずつの単純な分布(万分比)
    return pd.DataFrame(
        {
            "貯蓄現在高階級": ["100万円未満", "100～200万円", "200～300万円", "300万円以上"],
            "年": [2025] * 4,
            "世帯数分布": [2500, 2500, 2500, 2500],
        }
    )


class TestSavingsPercentile:
    def test_class_boundary(self, simple_dist):
        # 100万円ちょうど → 下位25%地点
        assert savings_percentile(simple_dist, 100) == pytest.approx(25.0)

    def test_interpolation_within_class(self, simple_dist):
        # 150万円 = 100〜200万円階級の中間 → 25% + 12.5%
        assert savings_percentile(simple_dist, 150) == pytest.approx(37.5)

    def test_zero_savings(self, simple_dist):
        assert savings_percentile(simple_dist, 0) == pytest.approx(0.0)

    def test_in_open_top_class_returns_lower_bound(self, simple_dist):
        # 上端が開いた階級では下限(=その階級より下の累積)を返す
        assert savings_percentile(simple_dist, 10000) == pytest.approx(75.0)

    def test_negative_raises(self, simple_dist):
        with pytest.raises(ValueError):
            savings_percentile(simple_dist, -1)

    def test_uses_latest_year_only(self, simple_dist):
        older = pd.DataFrame(
            {
                "貯蓄現在高階級": ["100万円未満", "100～200万円", "200～300万円", "300万円以上"],
                "年": [2020] * 4,
                "世帯数分布": [10000, 0, 0, 0],  # 混ざると結果が変わる分布
            }
        )
        combined = pd.concat([older, simple_dist], ignore_index=True)
        assert savings_percentile(combined, 150) == pytest.approx(37.5)

    def test_real_data_is_monotonic(self):
        dist = load_savings_distribution(DATA_DIR)
        p_low = savings_percentile(dist, 100)
        p_mid = savings_percentile(dist, 1200)
        p_high = savings_percentile(dist, 3000)
        assert 0 < p_low < p_mid < p_high < 100


class TestSavingsValueAt:
    def test_median_of_simple_dist(self, simple_dist):
        assert savings_value_at(simple_dist, 50) == pytest.approx(200.0)

    def test_roundtrip_with_percentile(self, simple_dist):
        value = savings_value_at(simple_dist, 37.5)
        assert savings_percentile(simple_dist, value) == pytest.approx(37.5)

    def test_out_of_range_raises(self, simple_dist):
        with pytest.raises(ValueError):
            savings_value_at(simple_dist, 120)

    def test_real_median_below_mean(self):
        # 貯蓄分布は右に歪むため、中央値 < 平均(直近1年の全世帯平均は約1900万円)
        dist = load_savings_distribution(DATA_DIR)
        median = savings_value_at(dist, 50)
        assert 500 < median < 1900
