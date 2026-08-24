"""ユーザー入力(万円単位)からの特徴量計算と入力検証。"""

# 家計調査「年間収入階級」の区分(下限値, 万円)。上限は次の階級の下限。
INCOME_CLASS_BOUNDS: list[tuple[str, float]] = [
    ("200万円未満", 0),
    ("200～250万円", 200),
    ("250～300万円", 250),
    ("300～350万円", 300),
    ("350～400万円", 350),
    ("400～450万円", 400),
    ("450～500万円", 450),
    ("500～550万円", 500),
    ("550～600万円", 550),
    ("600～650万円", 600),
    ("650～700万円", 650),
    ("700～750万円", 700),
    ("750～800万円", 750),
    ("800～900万円", 800),
    ("900～1000万円", 900),
    ("1000～1250万円", 1000),
    ("1250～1500万円", 1250),
    ("1500万円以上", 1500),
]


def income_class_of(annual_income_man: float) -> str:
    if annual_income_man <= 0:
        raise ValueError("年間収入は正の値を入力してください")

    matched = INCOME_CLASS_BOUNDS[0][0]
    for label, lower in INCOME_CLASS_BOUNDS:
        if annual_income_man >= lower:
            matched = label
    return matched


def compute_ratios(annual_income_man: float, savings_man: float, debt_man: float) -> dict:
    if annual_income_man <= 0:
        raise ValueError("年間収入は正の値を入力してください")
    if savings_man < 0 or debt_man < 0:
        raise ValueError("貯蓄・負債は0以上の値を入力してください")

    return {
        "貯蓄/年収倍率": savings_man / annual_income_man,
        "負債/年収倍率": debt_man / annual_income_man,
    }


def surplus_rate(monthly_disposable_man: float, monthly_expense_man: float) -> float:
    """黒字率(%) = (可処分所得 - 消費支出) / 可処分所得 * 100。"""
    if monthly_disposable_man <= 0:
        raise ValueError("手取り月収は正の値を入力してください")
    if monthly_expense_man < 0:
        raise ValueError("月間支出は0以上の値を入力してください")

    return (monthly_disposable_man - monthly_expense_man) / monthly_disposable_man * 100.0
