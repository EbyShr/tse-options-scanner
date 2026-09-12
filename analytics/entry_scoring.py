"""
ماژول الگوریتم امتیازدهی شفاف ۱۰ پیشنهاد برتر ورود (Top 10 Entry Suggestions)
با ۴ زیرامتیاز مجزا، فیلتر سخت، تولید خودکار دلایل و هشدار تمرکز سبد
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config import AppConfig

logger = logging.getLogger("OptionScanner.EntryScoring")


def calculate_dte_suitability_score(dte: float, opt_min: int = 10, opt_max: int = 40) -> float:
    """
    محاسبه امتیاز تناسب DTE بر اساس منحنی بهینه (بازه ۱۰ تا ۴۰ روز بیشترین امتیاز).
    سررسید خیلی نزدیک (ریسک گاما و انقضا) یا خیلی دور (کند بودن نوسان) امتیاز کمتری دارند.
    """
    if dte < 3:
        return 0.0
    elif dte < opt_min:
        # بین ۳ تا ۱۰ روز: شیب صعودی از ۳۰ تا ۹۰
        return round(30.0 + (dte - 3) * (60.0 / (opt_min - 3)), 1)
    elif opt_min <= dte <= opt_max:
        # بازه بهینه: ۱۰۰ کامل
        return 100.0
    elif dte <= 70:
        # بین ۴۰ تا ۷۰ روز: شیب ملایم نزولی از ۱۰۰ تا ۷۰
        return round(100.0 - (dte - opt_max) * (30.0 / (70 - opt_max)), 1)
    else:
        # بیش از ۷۰ روز: امتیاز ثابت ۵۵
        return 55.0


def calculate_underlying_readiness_score(row: pd.Series) -> Tuple[float, List[str]]:
    """
    محاسبه امتیاز آمادگی دارایی پایه بر اساس قواعد شفاف ۴ گانه:
    ۱. پولبک اخیر (افت ۲ تا ۴ روزه بعد از صعود مستند)
    ۲. بازگشت به بالای SMA5 یا نزدیکی به باند SMA20
    ۳. وضعیت صف خرید در آخرین روز جلسه معاملاتی
    ۴. خروج RSI از منطقه اشباع فروش
    
    خروجی: (امتیاز از ۱۰۰, فهرست دلایل فعال‌شده)
    """
    reasons = []
    opt_type = str(row.get("نوع قرارداد", ""))
    is_call = "Call" in opt_type or "خرید" in opt_type

    is_pullback = bool(row.get("پولبک پایه", False))
    dist_sma5 = float(row.get("فاصله پایه از SMA5 (%)", 0.0))
    dist_sma20 = float(row.get("فاصله پایه از SMA20 (%)", 0.0))
    queue = str(row.get("وضعیت صف پایه", "متعادل"))
    rsi = float(row.get("RSI14 پایه", 50.0))
    rsi_status = str(row.get("وضعیت RSI پایه", "عادی"))
    ret_1d = float(row.get("بازدهی ۱ روزه پایه (%)", 0.0))
    ret_5d = float(row.get("بازدهی ۵ روزه پایه (%)", 0.0))

    score_pullback = 0.0
    score_sma = 0.0
    score_queue = 0.0
    score_rsi = 0.0

    if is_call:
        # ۱. قاعده پولبک (حداکثر ۳۰ امتیاز)
        if is_pullback:
            score_pullback = 30.0
            reasons.append("دارایی پایه در پولبک سالم پس از صعود چندروزه")
        elif ret_5d > 2.0 and ret_1d > 0.5:
            score_pullback = 20.0
            reasons.append("روند صعودی پرقدرت میان‌مدت")
        elif ret_1d > 0:
            score_pullback = 12.0
        else:
            score_pullback = 5.0

        # ۲. قاعده حمایت و میانگین‌های متحرک SMA (حداکثر ۲۵ امتیاز)
        if dist_sma5 >= 0:
            score_sma = 25.0
            reasons.append("قیمت پایه تثبیت‌شده در بالای SMA5")
        elif abs(dist_sma20) <= 2.5:
            score_sma = 20.0
            reasons.append("قیمت پایه روی حمایت معتبر SMA20")
        elif dist_sma5 >= -1.5:
            score_sma = 12.0
        else:
            score_sma = 5.0

        # ۳. وضعیت صف خرید در روز پایانی (حداکثر ۲۵ امتیاز)
        if "صف خرید" in queue:
            score_queue = 25.0
            reasons.append("صف خرید قفل‌شده در دارایی پایه")
        elif "تقاضا" in queue:
            score_queue = 18.0
            reasons.append("برتری محسوس تقاضای خریداران در سهم پایه")
        elif "متعادل" in queue:
            score_queue = 10.0
        else:
            score_queue = 0.0

        # ۴. وضعیت شاخص RSI (حداکثر ۲۰ امتیاز)
        if "خروج از اشباع فروش" in rsi_status or (30 <= rsi <= 45 and ret_1d > 0):
            score_rsi = 20.0
            reasons.append(f"خروج RSI از اشباع فروش (مقدار: {rsi})")
        elif 45 < rsi <= 65:
            score_rsi = 15.0
            reasons.append(f"شاخص RSI در موقعیت پایدار ({rsi})")
        elif rsi > 70:
            score_rsi = 8.0  # احتیاط اشباع خرید
            reasons.append(f"RSI در محدوده اشباع خرید ({rsi})")
        else:
            score_rsi = 10.0

    else:
        # برای قراردادهای اختیار فروش (Put)
        if ret_5d < -2.0 or dist_sma5 < -1.0:
            score_pullback = 25.0
            reasons.append("روند نزولی پایه همسو با استراتژی Put")
        else:
            score_pullback = 10.0

        if dist_sma5 < 0:
            score_sma = 25.0
            reasons.append("سقوط قیمت زیر SMA5")
        else:
            score_sma = 10.0

        if "صف فروش" in queue:
            score_queue = 25.0
            reasons.append("صف فروش در دارایی پایه")
        elif "متعادل" in queue:
            score_queue = 12.0
        else:
            score_queue = 5.0

        if rsi >= 70 or (rsi >= 60 and ret_1d < 0):
            score_rsi = 20.0
            reasons.append(f"واگرایی یا اشباع خرید RSI سهم پایه ({rsi})")
        else:
            score_rsi = 10.0

    total_readiness = score_pullback + score_sma + score_queue + score_rsi
    return round(min(total_readiness, 100.0), 1), reasons


def generate_entry_explanation(
    sym: str,
    u_sym: str,
    liq_score: float,
    bub_score: float,
    read_score: float,
    dte_score: float,
    dte: int,
    val_today: float,
    ratio_5d: float,
    is_fomo: bool,
    reasons_list: List[str],
) -> str:
    """تولید یک خط متن خودکار و روان پیرامون دلیل رتبه کسب‌شده"""
    parts = []

    # بخش نقدینگی
    val_tomans = val_today / 10_000_000.0
    if val_tomans >= 1000:
        val_str = f"{round(val_tomans / 1000, 1)} میلیارد تومان"
    else:
        val_str = f"{round(val_tomans)} میلیون تومان"

    if is_fomo:
        parts.append(f"ارزش معامله بالا ({val_str}) با هشدار جهش فومو")
    elif liq_score >= 70:
        parts.append(f"نقدینگی بالا و فعال ({val_str} معامله امروز)")
    else:
        parts.append(f"نقدینگی قابل‌قبول ({val_str})")

    # بخش حباب
    if bub_score >= 75:
        parts.append("حباب قیمتی در ۲۰٪ ارزان‌ترین قراردادهای بازار امروز")
    elif bub_score >= 50:
        parts.append("قیمت منصفانه و نزدیک به ارزش تئوریک BSM")
    else:
        parts.append("دارای مقداری حباب قیمتی")

    # بخش آمادگی پایه
    if reasons_list:
        parts.append(reasons_list[0])
        if len(reasons_list) > 1 and len(parts) < 3:
            parts.append(reasons_list[1])

    # بخش سررسید
    if dte_score >= 95:
        parts.append(f"سررسید ایده‌آل ({dte} روز مانده)")
    else:
        parts.append(f"سررسید {dte} روزه")

    return "؛ ".join(parts) + "."


def compute_top_entry_suggestions(
    df_all: pd.DataFrame,
    config: AppConfig,
    top_n: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    فیلتر سخت و امتیازدهی روی قراردادهای واجد شرایط جهت استخراج ۱۰ پیشنهاد برتر ورود
    """
    if df_all.empty:
        return pd.DataFrame(), {"has_concentration_warning": False, "concentrated_symbol": None}

    logger.info("در حال اجرای الگوریتم ۱۰ پیشنهاد برتر ورود به معاملات...")

    top_cfg = config.top_entry
    hard_cfg = top_cfg

    # ۱. اعمال فیلتر سخت (Hard Filters)
    # - ارزش معامله امروز زیر آستانه (پیش‌فرض ۳۰۰ میلیون ریال)
    # - DTE کمتر از ۳ روز (ریسک انقضا)
    # - تعداد معاملات زیر آستانه (پیش‌فرض ۵ معامله)
    mask_hard = (
        (df_all["ارزش معاملات امروز (ریال)"] >= hard_cfg.min_trade_value_rials)
        & (df_all["روزهای تا سررسید (DTE)"] >= hard_cfg.min_dte)
        & (df_all["تعداد معاملات امروز"] >= hard_cfg.min_trade_count)
    )

    df_eligible = df_all[mask_hard].copy()
    logger.info(f"تعداد {len(df_eligible)} نماد پس از عبور از فیلترهای سخت وارد چرخه امتیازدهی شدند.")

    if df_eligible.empty:
        logger.warning("هیچ نمادی از فیلترهای سخت عبور نکرد.")
        return pd.DataFrame(), {"has_concentration_warning": False, "concentrated_symbol": None}

    n = len(df_eligible)

    # ۲. محاسبه زیرامتیازهای ۴ گانه
    # زیرامتیاز ۱: نقدینگی (رتبه صدکی ارزش معامله + پایداری با میانگین ۵ روزه)
    val_ranks = rankdata(df_eligible["ارزش معاملات امروز (ریال)"], method="average")
    pct_val = (val_ranks / n) * 100.0

    liq_subscores = []
    for idx, (_, r) in enumerate(df_eligible.iterrows()):
        p_val = pct_val[idx]
        ratio_5d = float(r.get("نسبت معامله به میانگین", 1.0))
        is_fomo = bool(r.get("جهش فومو", False))

        if is_fomo:
            stability_pts = 55.0  # جریمه جهش ناگهانی فومو
        elif 0.8 <= ratio_5d <= 3.0:
            stability_pts = 95.0
        elif ratio_5d < 0.5:
            stability_pts = 50.0
        else:
            stability_pts = 75.0

        sub_liq = 0.70 * p_val + 0.30 * stability_pts
        liq_subscores.append(round(sub_liq, 1))

    df_eligible["امتیاز نقدینگی"] = liq_subscores

    # زیرامتیاز ۲: ارزش نسبی / حباب (کمترین حباب نسبت به BSM بالاترین امتیاز را دارد)
    bub_ranks = rankdata(df_eligible["حباب خام (%)"], method="average")
    df_eligible["امتیاز ارزش نسبی (حباب)"] = np.round(100.0 - (bub_ranks / n) * 100.0, 1)

    # زیرامتیاز ۳: آمادگی دارایی پایه (قواعد تکنیکال: پولبک، SMA، صف، RSI)
    readiness_subscores = []
    readiness_reasons = []
    for _, r in df_eligible.iterrows():
        r_score, r_reasons = calculate_underlying_readiness_score(r)
        readiness_subscores.append(r_score)
        readiness_reasons.append(r_reasons)

    df_eligible["امتیاز آمادگی پایه"] = readiness_subscores

    # زیرامتیاز ۴: تناسب DTE (پنجره بهینه ۱۰ تا ۴۰ روز)
    dte_subscores = [
        calculate_dte_suitability_score(
            dte=float(r["روزهای تا سررسید (DTE)"]),
            opt_min=top_cfg.optimal_dte_min,
            opt_max=top_cfg.optimal_dte_max,
        )
        for _, r in df_eligible.iterrows()
    ]
    df_eligible["امتیاز تناسب DTE"] = dte_subscores

    # ۳. محاسبه امتیاز نهایی ترکیبی ورود
    w_l = top_cfg.weight_liquidity
    w_b = top_cfg.weight_relative_bubble
    w_r = top_cfg.weight_underlying_readiness
    w_d = top_cfg.weight_dte_suitability

    final_scores = (
        df_eligible["امتیاز نقدینگی"] * w_l
        + df_eligible["امتیاز ارزش نسبی (حباب)"] * w_b
        + df_eligible["امتیاز آمادگی پایه"] * w_r
        + df_eligible["امتیاز تناسب DTE"] * w_d
    )
    df_eligible["امتیاز نهایی ورود"] = np.round(final_scores, 1)

    # ۴. تولید متن خودکار "چرا این امتیاز"
    explanations = []
    for idx, (_, r) in enumerate(df_eligible.iterrows()):
        expl = generate_entry_explanation(
            sym=r["نماد"],
            u_sym=r["دارایی پایه"],
            liq_score=r["امتیاز نقدینگی"],
            bub_score=r["امتیاز ارزش نسبی (حباب)"],
            read_score=r["امتیاز آمادگی پایه"],
            dte_score=r["امتیاز تناسب DTE"],
            dte=int(r["روزهای تا سررسید (DTE)"]),
            val_today=float(r["ارزش معاملات امروز (ریال)"]),
            ratio_5d=float(r.get("نسبت معامله به میانگین", 1.0)),
            is_fomo=bool(r.get("جهش فومو", False)),
            reasons_list=readiness_reasons[idx],
        )
        explanations.append(expl)

    df_eligible["چرا این امتیاز"] = explanations

    # ۵. انتخاب ۱۰ مورد برتر
    df_top = (
        df_eligible.sort_values("امتیاز نهایی ورود", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # ۶. بررسی هشدار تمرکز پرتفوی (> 3 مورد روی یک دارایی پایه)
    concentration_info = {
        "has_concentration_warning": False,
        "concentrated_symbol": None,
        "count": 0,
    }

    if not df_top.empty:
        counts = df_top["دارایی پایه"].value_counts()
        for u_sym, cnt in counts.items():
            if cnt > top_cfg.max_same_underlying_alert:
                concentration_info["has_concentration_warning"] = True
                concentration_info["concentrated_symbol"] = u_sym
                concentration_info["count"] = int(cnt)
                break

    logger.info(f"الگوریتم انتخاب ۱۰ نماد برتر ورود با موفقیت به اتمام رسید (تعداد خروجی: {len(df_top)}).")
    return df_top, concentration_info
