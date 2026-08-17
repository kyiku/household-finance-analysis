"""一般用ミクロデータ(平成21年全国消費実態調査・十大費目)の読み込みと特徴量構築。

出典表記: 「一般用ミクロデータ(平成21年全国消費実態調査)」(総務省統計局)を加工して作成。
擬似データのため、分析結果は実証研究の結果とみなせない(教育・演習用)。

目的変数「貯蓄余力率」= 1 − 年間消費支出/年間収入(税込)。
家計調査の黒字率(可処分所得ベース)とは定義が異なる近似指標である点に注意。
特徴量には収入・支出の金額そのものを含めない(目的変数の構成要素でリークになるため)。
"""

from pathlib import Path

import pandas as pd

# 十大費目: 変数名 → 日本語名
EXPENSE_COLS: dict[str, str] = {
    "Food": "食料",
    "Housing": "住居",
    "LFW": "光熱・水道",
    "Furniture": "家具・家事用品",
    "Clothes": "被服及び履物",
    "Health": "保健医療",
    "Transport": "交通・通信",
    "Education": "教育",
    "Recreation": "教養娯楽",
    "OL_Expenditure": "その他消費",
}

# 世帯属性(すべて0/1に変換して使う)
ATTRIBUTE_COLS = ["3大都市圏", "3人以上世帯", "就業人員2人以上", "持家", "世帯主就業", "65歳以上"]


def load_microdata(micro_dir: str | Path, household: str = "z") -> pd.DataFrame:
    """CSV(冒頭5行は注記のためスキップ、Shift_JIS)を読み込む。household: z=全世帯, k=勤労者世帯。"""
    path = Path(micro_dir) / f"ippan_2009zensho_{household}_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"ミクロデータが見つかりません: {path}")
    return pd.read_csv(path, encoding="cp932", skiprows=5)


def load_model_results(data_dir: str | Path) -> dict | None:
    """train_saver_model.py が出力した結果CSVを読む。未生成なら None。"""
    data_dir = Path(data_dir)
    paths = {
        "metrics": data_dir / "microdata_model_metrics.csv",
        "importances": data_dir / "microdata_importances.csv",
        "group_shares": data_dir / "microdata_group_shares.csv",
    }
    if not all(p.exists() for p in paths.values()):
        return None
    return {
        "metrics": pd.read_csv(paths["metrics"]),
        "importances": pd.read_csv(paths["importances"]),
        "group_shares": pd.read_csv(paths["group_shares"], index_col=0),
    }


def build_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """特徴量X・目的変数y・メタ情報(貯蓄余力率, Weight)を返す。

    - y: 貯蓄余力率が中央値以上なら1(貯蓄余力の高い世帯)
    - X: 十大費目の支出構成比 + 世帯属性(0/1)のみ。金額そのものは含めない。
    """
    df = raw[(raw["Y_Income"] > 0) & (raw["L_Expenditure"] > 0)].reset_index(drop=True)

    annual_expense_yen = df["L_Expenditure"] * 12.0
    annual_income_yen = df["Y_Income"] * 1000.0
    margin_rate = 1.0 - annual_expense_yen / annual_income_yen

    shares = pd.DataFrame(
        {
            f"{jp}比率": df[en] / df["L_Expenditure"]
            for en, jp in EXPENSE_COLS.items()
        }
    )

    attributes = pd.DataFrame(
        {
            "3大都市圏": (df["3City"] == 1).astype(int),
            "3人以上世帯": (df["T_SeJinin"] == 3).astype(int),
            "就業人員2人以上": (df["T_SyuJinin"] == 2).astype(int),
            "持家": (df["T_JuSyoyu"] == 1).astype(int),
            "世帯主就業": (df["T_Syuhi"] == 1).astype(int),
            "65歳以上": (df["T_Age_65"] == 2).astype(int),
        }
    )

    X = pd.concat([shares, attributes], axis=1)
    meta = pd.DataFrame({"貯蓄余力率": margin_rate, "Weight": df["Weight"]})
    y = (margin_rate >= margin_rate.median()).astype(int).rename("貯蓄余力世帯")
    return X, y, meta
