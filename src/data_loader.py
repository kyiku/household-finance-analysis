"""家計調査CSVの読み込みと整形。すべて純粋関数(data_dir を引数に取る)。"""

from pathlib import Path

import pandas as pd

from src.periods import period_year

AGE_ORDER = ["29歳以下", "30～39歳", "40～49歳", "50～59歳", "60～69歳", "70歳以上"]
QUINTILE_ORDER = ["年収五分位1", "年収五分位2", "年収五分位3", "年収五分位4", "年収五分位5"]
INCOME_CLASS_ORDER = [
    "200万円未満", "200～250万円", "250～300万円", "300～350万円", "350～400万円",
    "400～450万円", "450～500万円", "500～550万円", "550～600万円", "600～650万円",
    "650～700万円", "700～750万円", "750～800万円", "800～900万円", "900～1000万円",
    "1000～1250万円", "1250～1500万円", "1500万円以上",
]


def _read_csv(data_dir: str | Path, filename: str) -> pd.DataFrame:
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {path}")
    return pd.read_csv(path)


def _pivot_wide(df: pd.DataFrame, index_cols: list[str]) -> pd.DataFrame:
    duplicated = df.duplicated(subset=[*index_cols, "項目"])
    if duplicated.any():
        raise ValueError(
            f"同一の区分・時期・項目に重複行が {int(duplicated.sum())} 件あります。データを確認してください。"
        )
    wide = (
        df.pivot_table(index=index_cols, columns="項目", values="値", aggfunc="mean")
        .reset_index()
        .rename_axis(columns=None)
    )
    return wide.assign(年=wide["時期"].map(period_year))


def load_savings_debt_by_income(data_dir: str | Path) -> pd.DataFrame:
    """年間収入階級×時期の 年間収入・貯蓄・負債(万円)+ 倍率。"""
    df = _read_csv(data_dir, "kakei_savings_debt_by_income.csv")
    wide = _pivot_wide(df, ["年間収入階級", "時期"])
    return wide.assign(
        **{
            "貯蓄/年収倍率": wide["貯蓄"] / wide["年間収入"],
            "負債/年収倍率": wide["負債"] / wide["年間収入"],
        }
    )


def load_savings_debt_by_age(data_dir: str | Path) -> pd.DataFrame:
    """年齢階級×時期の 年間収入・貯蓄・負債(万円)。"""
    df = _read_csv(data_dir, "kakei_savings_debt_by_age.csv")
    return _pivot_wide(df, ["年齢階級", "時期"])


def load_income_expense_by_age(data_dir: str | Path) -> pd.DataFrame:
    """年齢階級(5歳刻み)×時期の 実収入・消費支出・可処分所得・黒字・黒字率など。"""
    df = _read_csv(data_dir, "kakei_income_expense_by_age.csv")
    return _pivot_wide(df, ["年齢階級", "時期"])


def load_savings_breakdown_by_age(data_dir: str | Path) -> pd.DataFrame:
    """年齢階級×時期の貯蓄内訳(通貨性預貯金・定期性預貯金・生命保険など・有価証券・金融機関外)。"""
    df = _read_csv(data_dir, "kakei_savings_breakdown_by_age.csv")
    return _pivot_wide(df, ["年齢階級", "時期"])


def load_surplus_by_quintile(data_dir: str | Path) -> pd.DataFrame:
    """年収五分位×時期の 実収入・消費支出・黒字率・平均消費性向(勤労者世帯)。"""
    df = _read_csv(data_dir, "kakei_surplus_rate_by_income_quintile.csv")
    return _pivot_wide(df, ["年間収入五分位", "時期"])


def load_savings_distribution(data_dir: str | Path) -> pd.DataFrame:
    """貯蓄現在高階級別の世帯数分布(二人以上の世帯・年次、万分比)。"""
    df = _read_csv(data_dir, "kakei_savings_distribution.csv")
    for col in ("貯蓄現在高階級", "年", "世帯数分布"):
        if col not in df.columns:
            raise ValueError(f"kakei_savings_distribution.csv に列 {col} がありません")
    return df
