"""2019年全国家計構造調査の年齢×収入クロス表による「同年代×同収入」ベンチマーク。

注意: 家計調査とは別調査・別時点(2019年)のため、水準の直接比較はしないこと。
"""

from pathlib import Path

import pandas as pd

# アプリの年代6区分 → クロス表の年齢階級ラベル
CROSS_AGE_MAP: dict[str, str] = {
    "29歳以下": "30歳未満",
    "30～39歳": "30～39",
    "40～49歳": "40～49",
    "50～59歳": "50～59",
    "60～69歳": "60～69",
    "70歳以上": "70歳以上",
}

VALUE_COLS = ["年間収入額", "貯蓄現在高", "負債現在高", "世帯数分布"]

# 収入階級の表示順(100万円未満 → 50万円刻み → 2000万円以上)
KOUZOU_INCOME_ORDER = [
    "100万円未満",
    *[f"{lower}～{lower + 50}万円" for lower in range(100, 2000, 50)],
    "2000万円以上",
]


def kouzou_income_class_of(annual_income_man: float) -> str:
    """構造調査の年間収入階級(100万円未満/100〜2000万円の50万円刻み/2000万円以上)を返す。"""
    if annual_income_man <= 0:
        raise ValueError("年間収入は正の値を入力してください")
    if annual_income_man < 100:
        return "100万円未満"
    if annual_income_man >= 2000:
        return "2000万円以上"
    lower = int(annual_income_man // 50) * 50
    return f"{lower}～{lower + 50}万円"


def load_cross_benchmark(data_dir: str | Path) -> pd.DataFrame:
    """クロス表CSVを読み込み、(年齢階級, 年間収入階級)ごとの横持ちにして返す。"""
    path = Path(data_dir) / "kouzou_savings_by_age_income.csv"
    if not path.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {path}")
    df = pd.read_csv(path)
    wide = (
        df.pivot_table(index=["年齢階級", "年間収入階級"], columns="項目", values="値")
        .reset_index()
        .rename_axis(columns=None)
    )
    return wide


def cross_profile(
    wide: pd.DataFrame, age_bracket: str, annual_income_man: float
) -> pd.Series | None:
    """同年代×同収入階級のプロファイルを返す。該当セルが未公表なら None。"""
    if age_bracket not in CROSS_AGE_MAP:
        raise ValueError(f"未知の年代区分です: {age_bracket}")

    income_class = kouzou_income_class_of(annual_income_man)
    cell = wide[
        (wide["年齢階級"] == CROSS_AGE_MAP[age_bracket])
        & (wide["年間収入階級"] == income_class)
    ]
    if cell.empty or cell.iloc[0][["貯蓄現在高", "負債現在高"]].isna().any():
        return None
    return cell.iloc[0]
