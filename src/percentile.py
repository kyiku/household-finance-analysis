"""貯蓄現在高階級別の世帯数分布から、ユーザーの貯蓄のパーセンタイルを推定する。"""

import re

import pandas as pd

_UNDER = re.compile(r"^(\d+)万円未満$")
_RANGE = re.compile(r"^(\d+)～(\d+)万円$")
_OVER = re.compile(r"^(\d+)万円以上$")


def parse_class_bounds(label: str) -> tuple[float, float]:
    """階級ラベル(例: '1000～1200万円')を(下限, 上限)の万円タプルに変換する。"""
    if m := _UNDER.match(label):
        return 0.0, float(m.group(1))
    if m := _RANGE.match(label):
        return float(m.group(1)), float(m.group(2))
    if m := _OVER.match(label):
        return float(m.group(1)), float("inf")
    raise ValueError(f"解釈できない階級ラベルです: {label}")


def savings_percentile(dist_df: pd.DataFrame, savings_man: float) -> float:
    """貯蓄 savings_man(万円)が分布の下から何%地点かを返す(0〜100)。

    最新年の分布を使い、階級内は一様分布とみなして線形補間する。
    上端が開いた最上位階級では補間できないため、その階級より下の累積割合を返す
    (実際のパーセンタイルはそれ以上、の意味になる)。
    """
    if savings_man < 0:
        raise ValueError("貯蓄は0以上の値を入力してください")

    latest = dist_df[dist_df["年"] == dist_df["年"].max()]
    bounds = latest["貯蓄現在高階級"].map(parse_class_bounds)
    classes = (
        latest.assign(下限=bounds.map(lambda b: b[0]), 上限=bounds.map(lambda b: b[1]))
        .sort_values("下限")
        .reset_index(drop=True)
    )
    total = classes["世帯数分布"].sum()
    if total <= 0:
        raise ValueError("世帯数分布の合計が0です。データを確認してください")

    cumulative = 0.0
    for _, row in classes.iterrows():
        share = row["世帯数分布"] / total * 100.0
        if savings_man >= row["上限"]:
            cumulative += share
            continue
        if row["上限"] == float("inf"):
            return cumulative
        within = (savings_man - row["下限"]) / (row["上限"] - row["下限"])
        return cumulative + share * within
    return min(cumulative, 100.0)


def savings_value_at(dist_df: pd.DataFrame, pct: float) -> float:
    """分布の下から pct% 地点の貯蓄額(万円)を線形補間で返す(中央値なら pct=50)。

    上端が開いた最上位階級に落ちる場合は、その階級の下限を返す。
    """
    if not 0 <= pct <= 100:
        raise ValueError("pct は0〜100で指定してください")

    latest = dist_df[dist_df["年"] == dist_df["年"].max()]
    bounds = latest["貯蓄現在高階級"].map(parse_class_bounds)
    classes = (
        latest.assign(下限=bounds.map(lambda b: b[0]), 上限=bounds.map(lambda b: b[1]))
        .sort_values("下限")
        .reset_index(drop=True)
    )
    total = classes["世帯数分布"].sum()

    cumulative = 0.0
    for _, row in classes.iterrows():
        share = row["世帯数分布"] / total * 100.0
        if cumulative + share < pct:
            cumulative += share
            continue
        if row["上限"] == float("inf"):
            return float(row["下限"])
        within = (pct - cumulative) / share if share > 0 else 0.0
        return float(row["下限"] + (row["上限"] - row["下限"]) * within)
    return float(classes.iloc[-1]["下限"])
