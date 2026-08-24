"""家計調査ダッシュボード: 家計診断と「貯蓄できている人の特徴」分析。

設計判断の記録は DESIGN_DECISIONS.md を参照。
"""

import japanize_matplotlib  # noqa: F401  (matplotlib の日本語表示に必要)
import streamlit as st

from src.clustering import fit_household_types
from src.cross_benchmark import load_cross_benchmark
from src.microdata import load_model_results
from src.data_loader import (
    load_income_expense_by_age,
    load_savings_breakdown_by_age,
    load_savings_debt_by_age,
    load_savings_debt_by_income,
    load_savings_distribution,
    load_surplus_by_quintile,
)
from ui.diagnosis import render_diagnosis
from ui.insights import render_insights
from ui.trends import render_trends

DATA_DIR = "data"


@st.cache_data
def load_all_data(data_dir: str) -> dict:
    return {
        "by_income": load_savings_debt_by_income(data_dir),
        "by_age": load_savings_debt_by_age(data_dir),
        "income_expense": load_income_expense_by_age(data_dir),
        "breakdown": load_savings_breakdown_by_age(data_dir),
        "quintile": load_surplus_by_quintile(data_dir),
        "distribution": load_savings_distribution(data_dir),
        "cross": load_cross_benchmark(data_dir),
        "micro_results": load_model_results(data_dir),
    }


@st.cache_resource
def get_household_type_model(data_dir: str):
    return fit_household_types(load_savings_debt_by_income(data_dir))


def main() -> None:
    st.set_page_config(page_title="家計調査ダッシュボード", page_icon="💰", layout="wide")
    st.title("家計調査ダッシュボード")
    st.write(
        "総務省「家計調査」(2000〜2026年)と「2019年全国家計構造調査」をもとに、"
        "あなたの家計の立ち位置を分布・同年代×同収入などの視点で診断します。"
    )

    try:
        data = load_all_data(DATA_DIR)
        model = get_household_type_model(DATA_DIR)
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
        st.error(f"データの読み込み・前処理に失敗しました: {error}")
        st.stop()
        return

    tab1, tab2, tab3 = st.tabs(["🩺 あなたの家計診断", "💡 統計にみる貯蓄の傾向", "📈 データとトレンド"])
    with tab1:
        render_diagnosis(data, model)
    with tab2:
        render_insights(data)
    with tab3:
        render_trends(data)


main()
