"""「貯蓄できている人の特徴」を導くための集計。"""

import pandas as pd

from src.benchmark import latest_mean_by
from src.data_loader import AGE_ORDER, QUINTILE_ORDER

BREAKDOWN_COLS = ["通貨性預貯金", "定期性預貯金", "生命保険など", "有価証券", "金融機関外"]

# 収支データ(5歳刻み)の表示順。「34歳以下」は他階級と重複する集約区分のため除外。
FINE_AGE_ORDER = [
    "24歳以下", "25～29歳", "30～34歳", "35～39歳", "40～44歳",
    "45～49歳", "50～54歳", "55～59歳", "60～64歳", "65～69歳",
]

# 診断入力の粗い年代(貯蓄データ側) → 収支データ(5歳刻み)の年齢階級
COARSE_TO_FINE_AGE: dict[str, list[str] | None] = {
    "29歳以下": ["24歳以下", "25～29歳"],
    "30～39歳": ["30～34歳", "35～39歳"],
    "40～49歳": ["40～44歳", "45～49歳"],
    "50～59歳": ["50～54歳", "55～59歳"],
    "60～69歳": ["60～64歳", "65～69歳"],
    "70歳以上": None,  # 収支データに該当階級がない
}


def latest_surplus_stats_by_quintile(df_quintile: pd.DataFrame) -> pd.DataFrame:
    """直近1年(公表済みの最新4四半期)の年収五分位別 黒字率・平均消費性向・実収入。"""
    stats = latest_mean_by(
        df_quintile, group_col="年間収入五分位", value_cols=["黒字率", "平均消費性向", "実収入", "消費支出"]
    )
    return stats.reindex(QUINTILE_ORDER).round(1)


def yearly_mean_pivot(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    """年×グループの平均値ピボット(時系列グラフ用)。"""
    pivot = df.pivot_table(index="年", columns=group_col, values=value_col, aggfunc="mean")
    order = {"年間収入五分位": QUINTILE_ORDER, "年齢階級": AGE_ORDER}.get(group_col)
    if order is not None:
        pivot = pivot.reindex(columns=[c for c in order if c in pivot.columns])
    return pivot


def latest_breakdown_share_by_age(df_breakdown: pd.DataFrame) -> pd.DataFrame:
    """直近1年の年代別 貯蓄内訳の構成比(%)。行合計は100。"""
    latest = latest_mean_by(df_breakdown, group_col="年齢階級", value_cols=BREAKDOWN_COLS)
    totals = latest.sum(axis=1)
    share = latest.div(totals.where(totals > 0), axis=0) * 100.0
    return share.reindex(AGE_ORDER)


def coarse_age_surplus_rate(df_income_expense: pd.DataFrame, coarse_age: str) -> float | None:
    """粗い年代区分に対応する直近1年の平均黒字率(%)。対応データがなければ None。

    各5歳刻み階級について「その階級で公表済みの直近12か月」を平均するため、
    最新暦年に一部階級のデータが未公表でも(例: 年初時点の若年層)値を返せる。
    """
    if coarse_age not in COARSE_TO_FINE_AGE:
        raise ValueError(f"未知の年代区分です: {coarse_age}")

    fine_brackets = COARSE_TO_FINE_AGE[coarse_age]
    if fine_brackets is None:
        return None

    subset = df_income_expense[df_income_expense["年齢階級"].isin(fine_brackets)]
    if subset.empty:
        return None
    means = latest_mean_by(subset, group_col="年齢階級", value_cols=["黒字率"])
    return float(means["黒字率"].mean())
