"""タブ①: あなたの家計診断。"""

import matplotlib.pyplot as plt
import streamlit as st

from src.analysis import coarse_age_surplus_rate
from src.benchmark import build_comparison_table, latest_mean_by
from src.clustering import classify_user
from src.cross_benchmark import CROSS_AGE_MAP, cross_profile, kouzou_income_class_of
from src.data_loader import AGE_ORDER
from src.features import compute_ratios, income_class_of, surplus_rate
from src.percentile import parse_class_bounds, savings_percentile, savings_value_at


def _user_metrics(annual_income_man: float, savings_man: float, debt_man: float) -> dict:
    ratios = compute_ratios(annual_income_man, savings_man, debt_man)
    return {
        "年間収入(万円)": float(annual_income_man),
        "貯蓄(万円)": float(savings_man),
        "負債(万円)": float(debt_man),
        "貯蓄/年収倍率": ratios["貯蓄/年収倍率"],
        "負債/年収倍率": ratios["負債/年収倍率"],
    }


def _profile_metrics(row) -> dict:
    return {
        "年間収入(万円)": float(row["年間収入"]),
        "貯蓄(万円)": float(row["貯蓄"]),
        "負債(万円)": float(row["負債"]),
        "貯蓄/年収倍率": float(row["貯蓄"] / row["年間収入"]),
        "負債/年収倍率": float(row["負債"] / row["年間収入"]),
    }


def _render_percentile_section(data, savings_man):
    dist = data["distribution"]
    year = int(dist["年"].max())
    pct = savings_percentile(dist, savings_man)
    median = savings_value_at(dist, 50)

    st.subheader("① 貯蓄の立ち位置(二人以上世帯の分布)")
    col1, col2 = st.columns(2)
    col1.metric("あなたの位置", f"下から{pct:.0f}%地点", f"上位{100 - pct:.0f}%", delta_color="off")
    col2.metric(f"貯蓄の中央値({year}年)", f"約{median:,.0f}万円")
    st.caption(
        "貯蓄の分布は一部の高額世帯側に大きく歪んでいるため、平均との比較より"
        "「分布の中の位置」で見るのが適切です。階級内は一様とみなした概算値です。"
    )

    latest = dist[dist["年"] == year]
    ordered = latest.assign(下限=latest["貯蓄現在高階級"].map(lambda s: parse_class_bounds(s)[0])).sort_values("下限")
    shares = ordered["世帯数分布"] / ordered["世帯数分布"].sum() * 100.0
    user_idx = next(
        (i for i, label in enumerate(ordered["貯蓄現在高階級"])
         if parse_class_bounds(label)[0] <= savings_man < parse_class_bounds(label)[1]),
        len(ordered) - 1,
    )
    colors = ["tab:red" if i == user_idx else "tab:blue" for i in range(len(ordered))]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(range(len(ordered)), shares, color=colors)
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered["貯蓄現在高階級"], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("世帯の割合(%)")
    ax.set_title(f"貯蓄現在高の分布({year}年・二人以上世帯) — 赤があなたの階級")
    st.pyplot(fig)
    plt.close(fig)


def _render_cross_section(data, user, age_bracket, annual_income_man):
    income_class = kouzou_income_class_of(annual_income_man)
    cross_age = CROSS_AGE_MAP[age_bracket]
    st.subheader(f"② 同年代×同収入({cross_age}・{income_class})との比較")
    st.caption(
        "出典: 2019年全国家計構造調査(二人以上の世帯)。家計調査とは別の調査・別時点のため、"
        "③④の値と水準を直接比較しないでください。"
    )
    profile = cross_profile(data["cross"], age_bracket, annual_income_man)
    if profile is None:
        st.info(
            "この年代×収入の組み合わせは世帯数が少なく公表されていません。③④の比較を参照してください。"
        )
        return

    table = build_comparison_table(
        {
            "年間収入(万円)": user["年間収入(万円)"],
            "貯蓄(万円)": user["貯蓄(万円)"],
            "負債(万円)": user["負債(万円)"],
        },
        {
            "年間収入(万円)": float(profile["年間収入額"]),
            "貯蓄(万円)": float(profile["貯蓄現在高"]),
            "負債(万円)": float(profile["負債現在高"]),
        },
        f"{cross_age}×{income_class}平均(2019年)",
    )
    st.dataframe(table)
    diff = user["貯蓄(万円)"] - float(profile["貯蓄現在高"])
    st.metric("同年代×同収入の平均貯蓄との差", f"{diff:+,.0f}万円")


def _render_income_class_section(data, user, annual_income_man):
    income_class = income_class_of(annual_income_man)
    bench = latest_mean_by(data["by_income"], "年間収入階級", ["年間収入", "貯蓄", "負債"])
    st.subheader(f"③ 同じ収入階級({income_class})との比較")
    st.caption("ベンチマークは家計調査の直近1年(公表済みの最新4四半期)の二人以上世帯平均です。")
    profile = bench.loc[income_class]
    st.dataframe(build_comparison_table(user, _profile_metrics(profile), f"{income_class}平均"))
    diff = user["貯蓄(万円)"] - float(profile["貯蓄"])
    st.metric("同収入階級の平均貯蓄との差", f"{diff:+,.0f}万円")


def _render_age_section(data, user, age_bracket, savings_man):
    bench = latest_mean_by(data["by_age"], "年齢階級", ["年間収入", "貯蓄", "負債"])
    st.subheader(f"④ 同年代({age_bracket})との比較")
    st.caption("ベンチマークは家計調査の直近1年(公表済みの最新4四半期)の二人以上世帯平均です。")
    st.dataframe(
        build_comparison_table(user, _profile_metrics(bench.loc[age_bracket]), f"{age_bracket}平均")
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = bench.reindex(AGE_ORDER)
    ax.bar(range(len(ordered)), ordered["貯蓄"], width=0.5, label="年代平均の貯蓄(直近1年)")
    ax.axhline(savings_man, color="red", linestyle="--", label="あなたの貯蓄")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered.index, rotation=30, ha="right")
    ax.set_ylabel("万円")
    ax.set_title("年代別 平均貯蓄とあなたの位置")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def _render_surplus_section(data, age_bracket, monthly_disposable_man, monthly_expense_man):
    st.subheader("⑤ 黒字率診断(毎月貯蓄できているか)")
    if monthly_disposable_man <= 0:
        st.info("手取り月収と月間支出を入力すると、あなたの黒字率を同年代と比較できます。")
        return

    user_rate = surplus_rate(monthly_disposable_man, monthly_expense_man)
    bench_rate = coarse_age_surplus_rate(data["income_expense"], age_bracket)

    col1, col2 = st.columns(2)
    col1.metric("あなたの黒字率", f"{user_rate:.1f}%")
    if bench_rate is None:
        col2.metric(f"{age_bracket}の平均黒字率", "データなし")
        st.caption("この年代に対応する収支データ(勤労者世帯)が公表されていないため、比較値はありません。")
    else:
        col2.metric(
            f"{age_bracket}の平均黒字率(勤労者世帯)",
            f"{bench_rate:.1f}%",
            delta=f"{user_rate - bench_rate:+.1f}pt があなたとの差",
            delta_color="off",
        )
    st.caption("黒字率 = (可処分所得 − 消費支出) ÷ 可処分所得。家計調査では約3割が平均的な水準です。")


def _render_type_map_section(data, model, user, annual_income_man, savings_man, debt_man):
    st.subheader("⑥ 家計タイプマップ(参考)")
    st.warning(
        "この分類は世帯の個票ではなく「収入階級×四半期の集計値」(2002〜2025年)をKMeansで"
        "グルーピングしたものです。個々の世帯のばらつきは反映されないため、参考情報として見てください。"
    )
    type_name, profile = classify_user(model, annual_income_man, savings_man, debt_man)
    st.write(f"あなたに最も近いタイプ: **{type_name}**")
    st.dataframe(model.profile)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in model.assigned.groupby("家計タイプ"):
        ax.scatter(group["年間収入"], group["貯蓄"], alpha=0.4, s=15, label=name)
    ax.scatter([annual_income_man], [savings_man], color="red", s=180, marker="*", label="あなた", zorder=5)
    ax.set_xlabel("年間収入(万円)")
    ax.set_ylabel("貯蓄(万円)")
    ax.set_title("家計タイプマップ(集計値ベース)とあなたの位置")
    ax.legend(fontsize=8)
    st.pyplot(fig)
    plt.close(fig)


def render_diagnosis(data: dict, model) -> None:
    st.header("あなたの家計診断")
    st.write(
        "あなたの家計を「全世帯の分布の中の位置」「同年代×同収入」「同じ収入階級」「同じ年代」"
        "の4つの視点で診断します。"
    )

    col1, col2, col3 = st.columns(3)
    annual_income_man = col1.number_input("年間収入(万円)", min_value=1, value=600, step=10)
    savings_man = col2.number_input("貯蓄(万円)", min_value=0, value=1200, step=10)
    debt_man = col3.number_input("負債(万円)", min_value=0, value=500, step=10)
    age_bracket = st.selectbox("年代", AGE_ORDER, index=2)

    with st.expander("任意入力: 黒字率診断(毎月の収支)"):
        col4, col5 = st.columns(2)
        monthly_disposable_man = col4.number_input("手取り月収(万円)", min_value=0.0, value=0.0, step=1.0)
        monthly_expense_man = col5.number_input("月あたり消費支出(万円)", min_value=0.0, value=0.0, step=1.0)

    if not st.button("診断する"):
        return

    user = _user_metrics(annual_income_man, savings_man, debt_man)
    _render_percentile_section(data, savings_man)
    _render_cross_section(data, user, age_bracket, annual_income_man)
    _render_income_class_section(data, user, annual_income_man)
    _render_age_section(data, user, age_bracket, savings_man)
    _render_surplus_section(data, age_bracket, monthly_disposable_man, monthly_expense_man)
    _render_type_map_section(data, model, user, annual_income_man, savings_man, debt_man)
