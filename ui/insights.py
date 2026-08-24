"""タブ②: 貯蓄できている人の特徴(データセット全体からの知見)。"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.analysis import (
    BREAKDOWN_COLS,
    FINE_AGE_ORDER,
    latest_breakdown_share_by_age,
    latest_surplus_stats_by_quintile,
    yearly_mean_pivot,
)
from src.benchmark import latest_mean_by


def _render_quintile_section(df_quintile):
    stats = latest_surplus_stats_by_quintile(df_quintile)

    st.subheader("特徴1: 黒字率は収入階級でほぼ決まる")
    low = stats.iloc[0]["黒字率"]
    high = stats.iloc[-1]["黒字率"]
    st.write(
        f"直近1年の勤労者世帯では、黒字率(可処分所得のうち貯蓄に回せた割合)は "
        f"年収下位20%で **{low:.1f}%**、上位20%で **{high:.1f}%**。"
        f"収入が上がるほど支出の伸びは収入の伸びより小さく、貯蓄余力の差になっています。"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(range(len(stats)), stats["黒字率"], color="tab:blue")
    ax1.set_xticks(range(len(stats)))
    ax1.set_xticklabels([s.replace("年収五分位", "五分位") for s in stats.index], rotation=20)
    ax1.set_ylabel("%")
    ax1.set_title("黒字率(直近1年)")
    ax2.bar(range(len(stats)), stats["平均消費性向"], color="tab:orange")
    ax2.set_xticks(range(len(stats)))
    ax2.set_xticklabels([s.replace("年収五分位", "五分位") for s in stats.index], rotation=20)
    ax2.set_ylabel("%")
    ax2.set_title("平均消費性向(直近1年)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(stats)


def _render_quintile_trend(df_quintile):
    st.subheader("特徴2: どの収入層も黒字率は長期的に上昇している")
    pivot = yearly_mean_pivot(df_quintile, group_col="年間収入五分位", value_col="黒字率")
    st.write(
        "2000年代からの長期推移を見ると、全五分位で黒字率が上昇傾向にあります"
        "(特に2020年以降の上昇はコロナ禍での消費減の影響を含む点に注意)。"
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], label=col.replace("年収五分位", "五分位"))
    ax.set_ylabel("黒字率(%)")
    ax.set_title("年収五分位別 黒字率の推移(年平均)")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)


def _render_age_surplus_section(df_income_expense):
    st.subheader("特徴3: 黒字率のピークは収入が伸びる30代後半〜40代前半")
    bench = latest_mean_by(df_income_expense, "年齢階級", ["黒字率"]).reindex(FINE_AGE_ORDER)
    peak_bracket = bench["黒字率"].idxmax()
    st.write(
        f"直近1年の年齢階級別黒字率のピークは **{peak_bracket}({bench['黒字率'].max():.1f}%)**。"
        "単純な「年齢が上がるほど貯蓄できる」ではなく、収入が伸びる30代後半〜40代前半に貯蓄余力が"
        "最大になり、定年前後の60代で大きく低下する山型の構造です。"
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(bench)), bench["黒字率"], color="tab:green")
    ax.set_xticks(range(len(bench)))
    ax.set_xticklabels(bench.index, rotation=30, ha="right")
    ax.set_ylabel("黒字率(%)")
    ax.set_title("年齢階級別 黒字率(直近1年・勤労者世帯)")
    st.pyplot(fig)
    plt.close(fig)


def _render_breakdown_section(df_breakdown):
    share = latest_breakdown_share_by_age(df_breakdown)

    st.subheader("特徴4: 貯蓄の「中身」は年代で変わる")
    securities_young = share.loc["29歳以下", "有価証券"]
    securities_mid = share.loc["50～59歳", "有価証券"]
    time_deposit_young = share.loc["29歳以下", "定期性預貯金"]
    time_deposit_old = share.loc["70歳以上", "定期性預貯金"]
    st.write(
        "直近1年の貯蓄構成比では、若年層は通貨性預貯金(流動性資産)と有価証券の比率が高く"
        f"(有価証券は29歳以下で {securities_young:.1f}%、50代では {securities_mid:.1f}%)、"
        f"定期性預貯金の比率は年代とともに上がります({time_deposit_young:.1f}% → 70歳以上 {time_deposit_old:.1f}%)。"
        "教育・住宅資金を控える中年期に投資比率が下がるU字型で、近年の若年層は投資性資産の比率が高いのが特徴です。"
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = None
    for col in BREAKDOWN_COLS:
        ax.bar(range(len(share)), share[col], bottom=bottom, label=col)
        bottom = share[col] if bottom is None else bottom + share[col]
    ax.set_xticks(range(len(share)))
    ax.set_xticklabels(share.index, rotation=30, ha="right")
    ax.set_ylabel("構成比(%)")
    ax.set_title("年代別 貯蓄内訳の構成比(直近1年)")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_microdata_section(results):
    st.subheader("個票分析(機械学習): 貯蓄余力の高い世帯は何が違うか")
    st.caption(
        "「一般用ミクロデータ(平成21年全国消費実態調査)」(総務省統計局)を加工して作成。"
        "擬似的な個票データ(約4.6万世帯)による教育・演習用の分析であり、実証研究の結果とはみなせません。"
        "貯蓄余力率 = 1 − 年間消費支出/年間収入(税込)で定義し、中央値以上を「貯蓄余力の高い世帯」としています。"
    )
    if results is None:
        st.info("モデル結果が未生成です。`python scripts/train_saver_model.py` を実行してください。")
        return

    metrics = results["metrics"]
    auc = metrics["ROC-AUC"].max()
    st.write(
        f"支出の**構成比**(十大費目)と世帯属性だけを特徴量に、貯蓄余力の高い世帯を分類しました"
        f"(収入・支出の金額そのものは目的変数の構成要素のため特徴量から除外)。"
        f"最良モデル(ランダムフォレスト)の **ROC-AUC は {auc:.2f}** — 「何にお金を使っているか」だけで"
        f"貯蓄余力の高低がかなり識別できます。"
    )
    st.dataframe(metrics, hide_index=True)

    importances = results["importances"].head(8).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(importances["特徴量"], importances["重要度"], color="tab:purple")
    ax.set_title("何が貯蓄余力を分けるか(ランダムフォレストの特徴量重要度・上位8)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    shares = results["group_shares"]
    x = np.arange(len(shares.columns))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, shares.iloc[0], width=0.4, label=shares.index[0], color="tab:gray")
    ax.bar(x + 0.2, shares.iloc[1], width=0.4, label=shares.index[1], color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("比率", "") for c in shares.columns], rotation=30, ha="right")
    ax.set_ylabel("支出構成比(%)")
    ax.set_title("支出の中身の違い(加重平均)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    housing_low = shares.iloc[1]["住居比率"]
    housing_high = shares.iloc[0]["住居比率"]
    st.markdown(
        f"- 貯蓄余力の高い世帯は**住居費比率が低い**({housing_high:.1f}% → {housing_low:.1f}%。持家・低家賃)。"
        "**交通・通信比率も低い**\n"
        "- 食料・光熱の構成比が高いのは「消費全体を絞ると必需品の割合が相対的に上がる」ため\n"
        "- 世帯属性では**世帯主の就業**が重要度上位 — 収入額を特徴量に入れなくても働き方の情報が効く"
    )


def render_insights(data: dict) -> None:
    st.header("統計にみる貯蓄の傾向")
    st.write(
        "総務省「家計調査」の集計データから、貯蓄をめぐる階級・年代レベルの傾向を整理します。"
        "集計値に基づく傾向のため「個々の世帯・人の特徴」までは言えません。"
        "個々の世帯レベルの分析は、冒頭の個票(ミクロデータ)分析を参照してください。"
    )
    _render_microdata_section(data.get("micro_results"))
    _render_quintile_section(data["quintile"])
    _render_quintile_trend(data["quintile"])
    _render_age_surplus_section(data["income_expense"])
    _render_breakdown_section(data["breakdown"])

    st.subheader("まとめ")
    st.markdown(
        "- **収入水準**が黒字率の最大の説明要因(高収入層ほど消費性向が低い)\n"
        "- ライフステージの影響が大きく、**黒字率は30代後半〜40代前半がピーク**の山型(60代で大きく低下)\n"
        "- 資産構成は年代で大きく異なり、**若年層は投資性資産、高年代は定期性預貯金**の比率が高い\n"
        "- どの層も黒字率は長期上昇傾向(直近はコロナ禍の消費減の影響に注意)"
    )
