"""ベンチマークの算出と比較表の組み立て。

ベンチマークは「各グループで公表済みの直近1年分(四半期データは4期、月次データは12か月)」
の平均を使う。最新の暦年でフィルタすると、年途中のデータ(例: 2026年Q1のみ)や
グループ間の公表時期のずれで、季節バイアスやNaNが生じるため(DESIGN_DECISIONS.md D2)。
"""

from collections.abc import Mapping

import pandas as pd

from src.periods import parse_period_code


def latest_year(df: pd.DataFrame) -> int:
    return int(df["年"].max())


def annual_period_count(df: pd.DataFrame) -> int:
    """時期コードから1年あたりの期数を推定する(月次=12、四半期=4)。"""
    period = parse_period_code(int(df["時期"].iloc[0]))
    return 12 if period.start_month == period.end_month else 4


def latest_mean_by(
    df: pd.DataFrame, group_col: str, value_cols: list[str], n_periods: int | None = None
) -> pd.DataFrame:
    """グループごとに「そのグループの直近 n_periods 期」の平均を返す。

    n_periods を省略すると直近1年分(四半期なら4、月次なら12)を使う。
    """
    n = n_periods if n_periods is not None else annual_period_count(df)
    recent = df.sort_values("時期", ascending=False).groupby(group_col, sort=True).head(n)
    return recent.groupby(group_col)[value_cols].mean()


def build_comparison_table(
    user: Mapping[str, float], profile: Mapping[str, float], benchmark_label: str
) -> pd.DataFrame:
    """「あなた vs ベンチマーク」の比較表。user のキー順で行を並べる。"""
    rows = list(user.keys())
    return pd.DataFrame(
        {
            "あなた": [user[k] for k in rows],
            benchmark_label: [profile[k] for k in rows],
        },
        index=rows,
    ).round(2)
