"""e-Stat API から診断強化用の2データセットを取得して data/ に保存する。

1. kakei_savings_distribution.csv
   家計調査 貯蓄・負債編(0002210024) 貯蓄現在高階級別の世帯数分布(二人以上の世帯・年次)
   → パーセンタイル診断に使用

2. kouzou_savings_by_age_income.csv
   2019年全国家計構造調査(0003443480) 世帯主の年齢階級×年間収入階級別の
   貯蓄・負債現在高など(二人以上の世帯・全世帯・男女平均)
   → 「同年代×同収入」ベンチマークに使用

実行: ESTAT_APP_ID を環境変数か ../.env に設定して `python scripts/fetch_benchmark_data.py`
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DIST_STATS_ID = "0002210024"
CROSS_STATS_ID = "0003443480"

# 家計調査: 貯蓄現在高階級コード(cat03) 002〜020 = 100万円未満〜4000万円以上
DIST_CLASS_CODES = [f"{i:03d}" for i in range(2, 21)]

# 構造調査: 10歳階級2(301〜306)+（再掲）70歳以上 → アプリの年代6区分に対応
CROSS_AGE_CODES = ["301", "302", "303", "304", "305", "R03"]
CROSS_INCOME_CODES = ["00", *[f"{i:02d}" for i in range(1, 41)]]  # 平均 + 44区分中の非再掲40区分
CROSS_ITEM_CODES = {
    "03-2019": "年間収入額",
    "17-2019": "貯蓄現在高",
    "75-2019": "負債現在高",
    "07-2019": "世帯数分布",
    "06-2019": "集計世帯数",
}


def load_app_id() -> str:
    app_id = os.environ.get("ESTAT_APP_ID", "")
    if not app_id:
        env_path = Path(__file__).resolve().parent.parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ESTAT_APP_ID="):
                    app_id = line.split("=", 1)[1].strip()
    if not app_id:
        raise RuntimeError("ESTAT_APP_ID が未設定です(.env か環境変数で設定してください)")
    return app_id


def get_stats_data(app_id: str, stats_data_id: str, **filters) -> list[dict]:
    """getStatsData を呼び、VALUE のリスト(メタ名解決済み)を返す。"""
    params = {"appId": app_id, "statsDataId": stats_data_id, "metaGetFlg": "Y", **filters}
    res = requests.get(f"{BASE_URL}/getStatsData", params=params, timeout=60)
    res.raise_for_status()
    body = res.json()["GET_STATS_DATA"]
    result = body["RESULT"]
    if result["STATUS"] != 0:
        raise RuntimeError(f"e-Stat APIエラー: {result.get('ERROR_MSG')}")

    stat_data = body["STATISTICAL_DATA"]
    values = stat_data["DATA_INF"]["VALUE"]
    if isinstance(values, dict):
        values = [values]

    # コード→名称の対応表を組み立てる
    name_maps: dict[str, dict[str, str]] = {}
    for obj in stat_data["CLASS_INF"]["CLASS_OBJ"]:
        cls = obj["CLASS"]
        if isinstance(cls, dict):
            cls = [cls]
        name_maps[obj["@id"]] = {c["@code"]: c["@name"] for c in cls}

    rows = []
    for v in values:
        row = {"値": v["$"], "単位": v.get("@unit", "")}
        for axis, mapping in name_maps.items():
            code = v.get(f"@{axis}")
            if code is not None:
                row[axis] = mapping.get(code, code)
                row[f"{axis}_code"] = code
        rows.append(row)
    return rows


def fetch_savings_distribution(app_id: str) -> pd.DataFrame:
    rows = get_stats_data(
        app_id,
        DIST_STATS_ID,
        cdCat01="001",  # 世帯数分布(抽出率調整)
        cdCat02="03",  # 二人以上の世帯(2000年〜)
        cdCat03=",".join(DIST_CLASS_CODES),
    )
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "貯蓄現在高階級": df["cat03"].str.replace("貯蓄現在高階級", "", regex=False),
            "年": df["time"].str.replace("年", "", regex=False).astype(int),
            "世帯数分布": pd.to_numeric(df["値"], errors="coerce"),
        }
    ).dropna()
    return out.sort_values(["年", "貯蓄現在高階級"]).reset_index(drop=True)


def fetch_cross_benchmark(app_id: str) -> pd.DataFrame:
    rows = get_stats_data(
        app_id,
        CROSS_STATS_ID,
        cdTab=",".join(CROSS_ITEM_CODES),
        cdCat01="1",  # 二人以上の世帯
        cdCat02="0",  # 全世帯
        cdCat03="0",  # 男女平均
        cdCat04=",".join(CROSS_INCOME_CODES),
        cdCat05=",".join(CROSS_AGE_CODES),
    )
    df = pd.DataFrame(rows)
    out = pd.DataFrame(
        {
            "項目": df["tab_code"].map(CROSS_ITEM_CODES),
            "年齢階級": df["cat05"]
            .str.replace("（10歳階級2）", "", regex=False)
            .str.replace("（再掲）", "", regex=False),
            "年間収入階級": df["cat04"],
            "値": pd.to_numeric(df["値"], errors="coerce"),
            "単位": df["単位"],
        }
    ).dropna(subset=["値"])
    # 千円 → 万円 に統一(既存CSVと単位を揃える)
    is_sen_yen = out["単位"] == "千円"
    out.loc[is_sen_yen, "値"] = out.loc[is_sen_yen, "値"] / 10.0
    out.loc[is_sen_yen, "単位"] = "万円"
    return out.reset_index(drop=True)


def main() -> None:
    app_id = load_app_id()
    DATA_DIR.mkdir(exist_ok=True)

    dist = fetch_savings_distribution(app_id)
    dist_path = DATA_DIR / "kakei_savings_distribution.csv"
    dist.to_csv(dist_path, index=False)
    print(f"保存: {dist_path} ({len(dist)}行, {dist['年'].min()}〜{dist['年'].max()}年)")

    cross = fetch_cross_benchmark(app_id)
    cross_path = DATA_DIR / "kouzou_savings_by_age_income.csv"
    cross.to_csv(cross_path, index=False)
    print(f"保存: {cross_path} ({len(cross)}行)")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, requests.RequestException) as error:
        print(f"取得に失敗しました: {error}", file=sys.stderr)
        sys.exit(1)
