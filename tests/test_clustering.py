import numpy as np
import pandas as pd
import pytest

from src.clustering import FEATURE_COLS, classify_user, fit_household_types


@pytest.fixture
def synthetic_wide():
    rng = np.random.default_rng(0)
    groups = [
        # (収入中心, 貯蓄中心, 負債中心) — 4つの離れた塊
        (300, 200, 100),
        (300, 1500, 100),
        (1200, 500, 800),
        (1200, 4000, 200),
    ]
    rows = []
    for income_c, savings_c, debt_c in groups:
        for _ in range(30):
            income = income_c + rng.normal(0, 10)
            savings = savings_c + rng.normal(0, 20)
            debt = max(debt_c + rng.normal(0, 10), 0.0)
            rows.append(
                {
                    "年間収入": income,
                    "貯蓄": savings,
                    "負債": debt,
                    "貯蓄/年収倍率": savings / income,
                    "負債/年収倍率": debt / income,
                }
            )
    return pd.DataFrame(rows)


class TestFitHouseholdTypes:
    def test_assigns_named_types(self, synthetic_wide):
        model = fit_household_types(synthetic_wide, n_clusters=4, random_state=42)
        assert model.assigned["家計タイプ"].nunique() == 4
        assert set(model.profile.index) == set(model.assigned["家計タイプ"].unique())

    def test_type_names_are_unique_and_descriptive(self, synthetic_wide):
        model = fit_household_types(synthetic_wide, n_clusters=4, random_state=42)
        names = list(model.profile.index)
        assert len(names) == len(set(names))
        assert all(("収入" in n and "貯蓄" in n) for n in names)

    def test_does_not_mutate_input(self, synthetic_wide):
        before = synthetic_wide.copy()
        fit_household_types(synthetic_wide, n_clusters=4, random_state=42)
        pd.testing.assert_frame_equal(synthetic_wide, before)


class TestClassifyUser:
    def test_high_income_high_savings_user(self, synthetic_wide):
        model = fit_household_types(synthetic_wide, n_clusters=4, random_state=42)
        type_name, profile_row = classify_user(
            model, annual_income_man=1200, savings_man=4000, debt_man=200
        )
        assert type_name.startswith("高収入・高貯蓄")
        assert profile_row["サンプル数"] == 30

    def test_feature_cols_match(self):
        assert FEATURE_COLS == ["年間収入", "貯蓄", "負債", "貯蓄/年収倍率", "負債/年収倍率"]
