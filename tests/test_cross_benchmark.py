from pathlib import Path

import pytest

from src.cross_benchmark import (
    CROSS_AGE_MAP,
    cross_profile,
    kouzou_income_class_of,
    load_cross_benchmark,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class TestKouzouIncomeClassOf:
    def test_under_100(self):
        assert kouzou_income_class_of(80) == "100万円未満"

    def test_50man_step(self):
        assert kouzou_income_class_of(620) == "600～650万円"

    def test_lower_boundary_inclusive(self):
        assert kouzou_income_class_of(600) == "600～650万円"

    def test_top_class(self):
        assert kouzou_income_class_of(2500) == "2000万円以上"

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            kouzou_income_class_of(0)


class TestCrossAgeMap:
    def test_all_app_brackets_are_mapped(self):
        app_brackets = ["29歳以下", "30～39歳", "40～49歳", "50～59歳", "60～69歳", "70歳以上"]
        assert list(CROSS_AGE_MAP.keys()) == app_brackets


class TestLoadCrossBenchmark:
    def test_wide_columns(self):
        df = load_cross_benchmark(DATA_DIR)
        for col in ["年齢階級", "年間収入階級", "貯蓄現在高", "負債現在高", "年間収入額", "世帯数分布"]:
            assert col in df.columns

    def test_age_brackets(self):
        df = load_cross_benchmark(DATA_DIR)
        assert set(df["年齢階級"]) == {"30歳未満", "30～39", "40～49", "50～59", "60～69", "70歳以上"}


class TestCrossProfile:
    def test_typical_cell(self):
        df = load_cross_benchmark(DATA_DIR)
        profile = cross_profile(df, age_bracket="40～49歳", annual_income_man=620)
        assert profile is not None
        assert 0 < profile["貯蓄現在高"] < 3000
        # 全年代混在の平均(約1900万円)より大幅に低いはず(年齢調整の効果)
        assert profile["貯蓄現在高"] < 1500

    def test_missing_cell_returns_none(self):
        # 30歳未満×2000万円以上のような希少セルは公表されていない
        df = load_cross_benchmark(DATA_DIR)
        profile = cross_profile(df, age_bracket="29歳以下", annual_income_man=2500)
        assert profile is None

    def test_unknown_age_raises(self):
        df = load_cross_benchmark(DATA_DIR)
        with pytest.raises(ValueError):
            cross_profile(df, age_bracket="不明な年代", annual_income_man=500)
