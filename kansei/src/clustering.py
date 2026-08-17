"""家計タイプマップ(参考情報)。

注意: サンプルは世帯の個票ではなく「収入階級×四半期の集計値」。
アプリ側ではその旨を明示し、診断の主役にはしない(DESIGN_DECISIONS.md D4)。
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.features import compute_ratios

FEATURE_COLS = ["年間収入", "貯蓄", "負債", "貯蓄/年収倍率", "負債/年収倍率"]


@dataclass(frozen=True)
class HouseholdTypeModel:
    """st.cache_resource で全セッション共有されるため、assigned/profile は読み取り専用で扱うこと。"""

    scaler: StandardScaler
    kmeans: KMeans
    assigned: pd.DataFrame  # 元データ + 家計タイプ列
    profile: pd.DataFrame  # 家計タイプ別の平均値 + サンプル数


def _name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """クラスタ平均の中央値との比較で「高/低収入・高/低貯蓄型」の名前を付ける。"""
    income_median = profile["年間収入"].median()
    savings_median = profile["貯蓄/年収倍率"].median()

    names: dict[int, str] = {}
    used: dict[str, int] = {}
    for cluster_id, row in profile.iterrows():
        income_side = "高" if row["年間収入"] >= income_median else "低"
        savings_side = "高" if row["貯蓄/年収倍率"] >= savings_median else "低"
        base = f"{income_side}収入・{savings_side}貯蓄型"
        count = used.get(base, 0)
        used = {**used, base: count + 1}
        names[cluster_id] = base if count == 0 else f"{base}({count + 1})"
    return names


def fit_household_types(
    wide: pd.DataFrame, n_clusters: int = 4, random_state: int = 42
) -> HouseholdTypeModel:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(wide[FEATURE_COLS])

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = kmeans.fit_predict(x_scaled)

    labeled = wide.assign(クラスタ=labels)
    raw_profile = labeled.groupby("クラスタ")[FEATURE_COLS].mean()
    if set(raw_profile.index) != set(range(n_clusters)):
        # classify_user はクラスタ番号と profile の行順の一致に依存する
        raise RuntimeError("KMeansで空クラスタが発生しました。n_clusters か入力データを見直してください。")
    names = _name_clusters(raw_profile)

    assigned = labeled.assign(家計タイプ=labeled["クラスタ"].map(names)).drop(columns=["クラスタ"])
    profile = (
        raw_profile.assign(
            サンプル数=labeled.groupby("クラスタ").size(),
            家計タイプ=raw_profile.index.map(names),
        )
        .set_index("家計タイプ")
        .round(2)
    )

    return HouseholdTypeModel(scaler=scaler, kmeans=kmeans, assigned=assigned, profile=profile)


def classify_user(
    model: HouseholdTypeModel, annual_income_man: float, savings_man: float, debt_man: float
) -> tuple[str, pd.Series]:
    """ユーザー入力が最も近い家計タイプ(参考)とそのプロファイルを返す。"""
    ratios = compute_ratios(annual_income_man, savings_man, debt_man)
    x = pd.DataFrame(
        [[annual_income_man, savings_man, debt_man, ratios["貯蓄/年収倍率"], ratios["負債/年収倍率"]]],
        columns=FEATURE_COLS,
    )
    cluster_id = int(model.kmeans.predict(model.scaler.transform(x))[0])
    # fit 時の groupby("クラスタ") は 0..n-1 の昇順のため、profile の行順がクラスタ番号に対応する
    type_name = str(model.profile.index[cluster_id])
    return type_name, model.profile.loc[type_name]
