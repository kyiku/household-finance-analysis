from pathlib import Path

import pandas as pd
import pytest

from src.microdata import (
    ATTRIBUTE_COLS,
    EXPENSE_COLS,
    build_dataset,
    load_microdata,
    load_model_results,
)

MICRO_DIR = Path(__file__).resolve().parent.parent / "microdata" / "ippan_2009zensho"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def synthetic_raw():
    return pd.DataFrame(
        {
            "3City": [1, 0, 1, 0],
            "T_SeJinin": [2, 3, 2, 3],
            "T_SyuJinin": [1, 2, 1, 2],
            "T_JuSyoyu": [1, 2, 1, 2],
            "T_Syuhi": [1, 1, 2, 2],
            "T_Age_5s": [3, 5, 0, 0],
            "T_Age_65": [1, 1, 2, 2],
            "Weight": [100.0, 200.0, 150.0, 50.0],
            "Y_Income": [6000.0, 4800.0, 2400.0, 0.0],  # 千円/年(最後は不正データ)
            "L_Expenditure": [250000.0, 400000.0, 150000.0, 100000.0],  # 円/月
            "Food": [50000.0, 100000.0, 50000.0, 30000.0],
            "Housing": [50000.0, 50000.0, 10000.0, 10000.0],
            "LFW": [20000.0, 30000.0, 15000.0, 10000.0],
            "Furniture": [10000.0, 20000.0, 5000.0, 5000.0],
            "Clothes": [10000.0, 20000.0, 5000.0, 5000.0],
            "Health": [10000.0, 20000.0, 15000.0, 5000.0],
            "Transport": [40000.0, 60000.0, 20000.0, 15000.0],
            "Education": [20000.0, 50000.0, 0.0, 0.0],
            "Recreation": [20000.0, 30000.0, 15000.0, 10000.0],
            "OL_Expenditure": [20000.0, 20000.0, 15000.0, 10000.0],
        }
    )


class TestLoadMicrodata:
    def test_loads_real_file(self):
        df = load_microdata(MICRO_DIR)
        assert len(df) > 40000
        for col in ["Y_Income", "L_Expenditure", "Food", "Weight"]:
            assert col in df.columns

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_microdata(tmp_path)


class TestBuildDataset:
    def test_drops_nonpositive_income(self, synthetic_raw):
        X, y, meta = build_dataset(synthetic_raw)
        assert len(X) == 3  # Y_Income=0 の行が落ちる

    def test_saving_margin_rate(self, synthetic_raw):
        _, _, meta = build_dataset(synthetic_raw)
        # 1行目: 1 - (12*250000)/(6000*1000) = 0.5
        assert meta.iloc[0]["貯蓄余力率"] == pytest.approx(0.5)

    def test_expense_shares_sum_to_one(self, synthetic_raw):
        X, _, _ = build_dataset(synthetic_raw)
        share_cols = [f"{c}比率" for c in EXPENSE_COLS.values()]
        assert X[share_cols].sum(axis=1).round(6).eq(1.0).all()

    def test_no_leakage_features(self, synthetic_raw):
        X, _, _ = build_dataset(synthetic_raw)
        # 目的変数の構成要素(収入・支出額)は特徴量に含めない
        for banned in ["Y_Income", "L_Expenditure", "年間収入", "消費支出", "貯蓄余力率", "Weight"]:
            assert banned not in X.columns

    def test_target_is_binary_median_split(self, synthetic_raw):
        _, y, meta = build_dataset(synthetic_raw)
        assert set(y.unique()) <= {0, 1}
        assert (y == (meta["貯蓄余力率"] >= meta["貯蓄余力率"].median()).astype(int)).all()

    def test_attribute_features_present(self, synthetic_raw):
        X, _, _ = build_dataset(synthetic_raw)
        for col in ATTRIBUTE_COLS:
            assert col in X.columns

    def test_does_not_mutate_input(self, synthetic_raw):
        before = synthetic_raw.copy()
        build_dataset(synthetic_raw)
        pd.testing.assert_frame_equal(synthetic_raw, before)


class TestLoadModelResults:
    def test_loads_generated_results(self):
        results = load_model_results(DATA_DIR)
        assert results is not None
        assert set(results) == {"metrics", "importances", "group_shares"}
        assert "ROC-AUC" in results["metrics"].columns
        assert len(results["group_shares"]) == 2

    def test_missing_files_return_none(self, tmp_path):
        assert load_model_results(tmp_path) is None
