from decimal import ROUND_HALF_UP, Decimal, getcontext

round_context = getcontext()
round_context.rounding = ROUND_HALF_UP


def c_round(x: float, digits: int, precision: int = 10) -> float:
    tmp = round(Decimal(x), precision)
    return float(tmp.__round__(digits))
