"""
ماژول محاسبات ارزش‌گذاری بلک-شولز (BSM)، یونانی‌ها، اهرم و وضعیت سودآوری (Moneyness)
"""

import math
from typing import Tuple
import numpy as np
from scipy.stats import norm


def calculate_bsm_price(
    spot: float,
    strike: float,
    dte: float,
    rate: float = 0.28,
    volatility: float = 0.35,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> Tuple[float, float]:
    """
    محاسبه قیمت تئوریک بلک-شولز و دلتای قرارداد.
    :param spot: قیمت روز دارایی پایه
    :param strike: قیمت اعمال قرارداد
    :param dte: روزهای مانده تا سررسید (Days to Expiration)
    :param rate: نرخ سود بدون ریسک سالانه
    :param volatility: نوسان‌پذیری سالانه (ضمنی یا تاریخی)
    :param option_type: نوع قرارداد ('call' یا 'put')
    :param dividend_yield: سود نقدی دارایی پایه
    :return: (قیمت تئوریک BSM, دلتا)
    """
    if spot <= 0 or strike <= 0:
        return 0.0, 0.0

    T = max(dte / 365.0, 1e-5)
    vol = max(volatility, 0.05)
    is_call = option_type.lower() in ("call", "اختیار خرید", "خرید")

    # اگر در سررسید باشیم یا زمان بسیار ناچیز باشد (ارزش ذاتی خالص)
    if dte <= 0:
        intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))
        delta_val = (1.0 if spot > strike else 0.0) if is_call else (-1.0 if spot < strike else 0.0)
        return intrinsic, delta_val

    try:
        d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)

        df_q = np.exp(-dividend_yield * T)
        df_r = np.exp(-rate * T)

        if is_call:
            price = spot * df_q * norm.cdf(d1) - strike * df_r * norm.cdf(d2)
            delta = df_q * norm.cdf(d1)
        else:
            price = strike * df_r * norm.cdf(-d2) - spot * df_q * norm.cdf(-d1)
            delta = -df_q * norm.cdf(-d1)

        return float(max(0.0, price)), float(delta)

    except Exception:
        intrinsic = max(0.0, (spot - strike) if is_call else (strike - spot))
        return intrinsic, 0.0


def calculate_leverage(spot: float, option_price: float, delta: float) -> float:
    """
    محاسبه اهرم واقعی قرارداد:
    اهرم ≈ (قیمت دارایی پایه × |دلتا|) / قیمت قرارداد
    """
    if option_price <= 0 or spot <= 0:
        return 0.0

    raw_leverage = (spot * abs(delta)) / option_price
    # محدود کردن اعداد عجیب در قراردادهای بسیار ارزان
    return round(min(raw_leverage, 100.0), 2)


def evaluate_moneyness(spot: float, strike: float, option_price: float, option_type: str = "call") -> Tuple[str, float]:
    """
    تعیین وضعیت سودآوری قرارداد (Moneyness) و درصد فاصله تا نقطه سربه‌سر (Break-even Distance %).
    - ITM: در سود
    - ATM: بی‌تفاوت (در محدوده ±۲٪ قیمت اعمال)
    - OTM: در زیان
    """
    if spot <= 0 or strike <= 0:
        return "نامشخص", 0.0

    is_call = option_type.lower() in ("call", "اختیار خرید", "خرید")
    ratio = spot / strike

    # وضعیت سودآوری
    if is_call:
        if ratio > 1.02:
            status = "در سود (ITM)"
        elif ratio < 0.98:
            status = "در زیان (OTM)"
        else:
            status = "بی‌تفاوت (ATM)"

        # نقطه سربه‌سر خریدار کال = قیمت اعمال + قیمت پرداختی آپشن
        be_price = strike + max(0.0, option_price)
        dist_to_be_pct = ((be_price - spot) / spot) * 100.0
    else:
        if ratio < 0.98:
            status = "در سود (ITM)"
        elif ratio > 1.02:
            status = "در زیان (OTM)"
        else:
            status = "بی‌تفاوت (ATM)"

        # نقطه سربه‌سر خریدار پوت = قیمت اعمال - قیمت پرداختی آپشن
        be_price = max(0.0, strike - max(0.0, option_price))
        dist_to_be_pct = ((spot - be_price) / spot) * 100.0

    return status, round(dist_to_be_pct, 2)
