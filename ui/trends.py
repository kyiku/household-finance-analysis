"""タブ③: データとトレンド(出典・注意点・長期推移)。"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.analysis import yearly_mean_pivot
from src.benchmark import latest_mean_by
from src.cross_benchmark import CROSS_AGE_MAP, KOUZOU_INCOME_ORDER
from src.data_loader import INCOME_CLASS_ORDER

DATASET_NOTES = pd.DataFrame(
    {
        "ファイル": [
            "kakei_savings_debt_by_income.csv",
            "kakei_savings_debt_by_age.csv",
            "kakei_savings_breakdown_by_age.csv",
            "kakei_income_expense_by_age.csv",
            "kakei_surplus_rate_by_income_quintile.csv",
            "kakei_savings_distribution.csv",
            "kouzou_savings_by_age_income.csv",
        ],
        "内容": [
            "年間収入階級別(18階級)の年間収入・貯蓄・負債",
            "年齢階級別(6区分)の年間収入・貯蓄・負債",
            "年齢階級別の貯蓄内訳(預貯金・保険・有価証券など)",
            "年齢階級別(5歳刻み)の実収入・消費支出・黒字率など",
            "年収五分位別の実収入・消費支出・黒字率・平均消費性向",
            "貯蓄現在高階級別(19階級)の世帯数分布 — パーセンタイル診断用",
            "世帯主の年齢×年間収入階級別の貯蓄・負債 — 同年代×同収入比較用",
        ],
        "出典": [
            "家計調査", "家計調査", "家計調査", "家計調査", "家計調査",
            "家計調査", "2019年全国家計構造調査",
        ],
        "期間": [
            "2002〜2025年(四半期)", "2002〜2025年(四半期)", "2002〜2025年(四半期)",
            "2000〜2026年(月次)", "2000〜2026年(四半期)", "2002〜2025年(年次)", "2019年",
        ],
        "対象": [
            "二人以上世帯", "二人以上世帯", "二人以上世帯", "二人以上・勤労者世帯",
            "勤労者世帯", "二人以上世帯", "二人以上世帯",
        ],
    }
)


def _render_savings_debt_trend(df_age):
    st.subheader("年代別 貯蓄・負債の長期推移")
    savings = yearly_mean_pivot(df_age, group_col="年齢階級", value_col="貯蓄")
    debt = yearly_mean_pivot(df_age, group_col="年齢階級", value_col="負債")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for col in savings.columns:
        ax1.plot(savings.index, savings[col], label=col)
        ax2.plot(debt.index, debt[col], label=col)
    ax1.set_title("平均貯蓄(万円)")
    ax2.set_title("平均負債(万円)")
    ax1.legend(fontsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "高年代ほど貯蓄が積み上がる一方、負債(主に住宅ローン)は30〜40代に集中する構造が"
        "20年以上一貫しています。近年は30〜40代の負債水準の上昇(住宅価格高騰)が目立ちます。"
    )


def _render_income_class_snapshot(df_income):
    st.subheader("収入階級別 貯蓄・負債(直近1年)")
    grouped = latest_mean_by(df_income, group_col="年間収入階級", value_cols=["年間収入", "貯蓄", "負債"])
    ordered = grouped.reindex([c for c in INCOME_CLASS_ORDER if c in grouped.index])

    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(ordered))
    ax.bar([i - 0.2 for i in x], ordered["貯蓄"], width=0.4, label="貯蓄")
    ax.bar([i + 0.2 for i in x], ordered["負債"], width=0.4, label="負債")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ordered.index, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("万円")
    ax.set_title("収入階級別 平均貯蓄・負債(直近1年)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "注意: 低収入階級には年金生活の高齢世帯(収入は少なくても現役時代の貯蓄を持つ)が"
        "多く含まれるため、左端でも平均貯蓄が高く見えます。年齢の影響を除いた比較は下のグラフを参照。"
    )


def _render_cross_income_chart(df_cross):
    st.subheader("年代で絞った 収入階級別 貯蓄・負債(2019年)")
    st.caption(
        "出典: 2019年全国家計構造調査(二人以上の世帯)。上のグラフ(家計調査)とは別の調査のため、"
        "水準の直接比較はできません。年代を固定すると収入と貯蓄の素直な関係が見えます。"
    )
    age = st.selectbox("年代", list(CROSS_AGE_MAP.values()), index=2, key="cross_chart_age")
    sub = df_cross[df_cross["年齢階級"] == age].set_index("年間収入階級")
    ordered = sub.reindex([c for c in KOUZOU_INCOME_ORDER if c in sub.index]).dropna(
        subset=["貯蓄現在高"]
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(len(ordered))
    ax.bar([i - 0.2 for i in x], ordered["貯蓄現在高"], width=0.4, label="貯蓄現在高")
    ax.bar([i + 0.2 for i in x], ordered["負債現在高"], width=0.4, label="負債現在高")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ordered.index, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("万円")
    ax.set_title(f"{age}の世帯: 収入階級別 平均貯蓄・負債(2019年)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_trends(data: dict) -> None:
    st.header("データとトレンド")
    st.write("出典: 総務省統計局「家計調査」(e-Stat API 経由で取得)。")
    st.dataframe(DATASET_NOTES, hide_index=True)
    st.caption(
        "注意: いずれも標本調査の集計値です。「二人以上世帯」と「勤労者世帯」で対象が異なるため、"
        "タブ間で水準を直接比較しないでください。黒字率は勤労者世帯のみ公表されています。"
    )
    _render_savings_debt_trend(data["by_age"])
    _render_income_class_snapshot(data["by_income"])
    _render_cross_income_chart(data["cross"])
