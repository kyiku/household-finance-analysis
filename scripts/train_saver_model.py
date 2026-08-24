"""一般用ミクロデータで「貯蓄余力の高い世帯」分類モデルを学習し、結果を data/ に保存する。

学習はこのスクリプトで事前に行い、アプリは結果CSVを読むだけにする(起動を軽く保つため)。
実行: `python scripts/train_saver_model.py` (リポジトリ直下から)

出力:
- data/microdata_model_metrics.csv   … モデル別の精度・ROC-AUC
- data/microdata_importances.csv     … RandomForest の特徴量重要度
- data/microdata_group_shares.csv    … 貯蓄余力世帯/非貯蓄余力世帯の支出構成比(加重平均)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.microdata import EXPENSE_COLS, build_dataset, load_microdata  # noqa: E402

MICRO_DIR = Path(__file__).resolve().parent.parent / "microdata" / "ippan_2009zensho"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 42


def build_models() -> dict:
    return {
        "ロジスティック回帰": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "決定木(深さ4)": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "ランダムフォレスト": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }


def main() -> None:
    raw = load_microdata(MICRO_DIR, household="z")
    X, y, meta = build_dataset(raw)
    weight = meta["Weight"]

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weight, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    metrics_rows = []
    rf_model = None
    for name, model in build_models().items():
        fit_params = {}
        last_step = model.steps[-1][0] if hasattr(model, "steps") else None
        if last_step is not None:
            fit_params[f"{last_step}__sample_weight"] = w_train
        else:
            fit_params["sample_weight"] = w_train
        model.fit(X_train, y_train, **fit_params)

        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        metrics_rows.append(
            {
                "モデル": name,
                "精度": accuracy_score(y_test, pred, sample_weight=w_test),
                "ROC-AUC": roc_auc_score(y_test, proba, sample_weight=w_test),
            }
        )
        if name == "ランダムフォレスト":
            rf_model = model

    metrics = pd.DataFrame(metrics_rows).round(3)
    metrics.to_csv(DATA_DIR / "microdata_model_metrics.csv", index=False)

    importances = (
        pd.DataFrame({"特徴量": X.columns, "重要度": rf_model.feature_importances_})
        .sort_values("重要度", ascending=False)
        .round(4)
    )
    importances.to_csv(DATA_DIR / "microdata_importances.csv", index=False)

    share_cols = [f"{jp}比率" for jp in EXPENSE_COLS.values()]
    grouped = pd.concat([X[share_cols], y], axis=1)
    group_shares = (
        grouped.groupby("貯蓄余力世帯")[share_cols]
        .apply(lambda g: pd.Series(np.average(g, axis=0, weights=weight.loc[g.index]), index=share_cols))
        .rename(index={0: "貯蓄余力の低い世帯", 1: "貯蓄余力の高い世帯"})
        .mul(100.0)
        .round(2)
    )
    group_shares.to_csv(DATA_DIR / "microdata_group_shares.csv")

    print(f"世帯数: {len(X):,} (学習{len(X_train):,} / 検証{len(X_test):,})")
    print(metrics.to_string(index=False))
    print("\n重要度トップ5:")
    print(importances.head(5).to_string(index=False))
    print("\n支出構成比(加重平均, %):")
    print(group_shares.to_string())


if __name__ == "__main__":
    main()
