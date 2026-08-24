"""e-Stat の時期コード(例: 2025001012 = 2025年10〜12月期)のパース。

コード体系: YYYY + "00" + 開始月2桁 + 終了月2桁。月次データは開始月=終了月。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Period:
    year: int
    start_month: int
    end_month: int


def parse_period_code(code: int) -> Period:
    text = str(int(code))
    if len(text) != 10:
        raise ValueError(f"時期コードは10桁である必要があります: {code}")

    year = int(text[:4])
    start_month = int(text[6:8])
    end_month = int(text[8:10])

    if not (1 <= start_month <= 12 and 1 <= end_month <= 12 and start_month <= end_month):
        raise ValueError(f"時期コードの月が不正です: {code}")

    return Period(year=year, start_month=start_month, end_month=end_month)


def period_year(code: int) -> int:
    return parse_period_code(code).year


def period_label(code: int) -> str:
    period = parse_period_code(code)
    if period.start_month == period.end_month:
        return f"{period.year}年{period.start_month}月"
    return f"{period.year}年{period.start_month}-{period.end_month}月期"
