"""
ماژول الگوریتم امتیازدهی واحد، جامع و نامتقارن اسکنر آپشن بورس تهران (Unified Scoring System)
نسخه نهایی پچ v2 (اصلاح وزن‌دهی و سقف تنوع نماد):
- وزن‌های ۲۵/۳۰/۳۰/۱۵ (آمادگی پایه ۲۵٪، نقدینگی ۳۰٪، ارزش نسبی ۳۰٪، تناسب سررسید ۱۵٪)
- سقف تنوع دارایی پایه در انتخاب نهایی Top 10 (select_top_n_diversified با پیش‌فرض ۳)
- فرمول دقیق سقف‌دار جهش نقدینگی: min(momentum_ratio / 3.0, 1.0) * 100
- ایمن‌سازی کامل برابر داده‌های NaN/None در BSM و دلتا
- تفکیک آستانه فیلتر سخت Call (۵۰۰ م.ت و ۳۰ معامله) و Put (۱۰۰ م.ت و ۱۰ معامله)
- منطق نامتقارن ۴ مؤلفه‌ای آمادگی پایه (میانگین وزنی مساوی ۲۵٪ از ۱۰۰)
- سقف ضد-Chasing با فالبک هوشمند ۱ روزه (آستانه ۵٪)
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config import AppConfig
from data.normalizer import normalize_fa, symbols_match

logger = logging.getLogger("OptionScanner.UnifiedScoring")


def calculate_dte_suitability_score(dte: float, opt_min: int = 10, opt_max: int = 40) -> float:
    """
    محاسبه امتیاز تناسب DTE بر اساس منحنی پیوسته (۱۰ تا ۴۰ روز بالاترین امتیاز).
    کمتر از ۳ روز صفر، ۳ تا ۵ روز شیب تند، و بیش از ۶۰ روز جریمه تدریجی می‌شوند.
    """
    if dte < 3:
        return 0.0
    elif dte < 5:
        # بین ۳ تا ۵ روز: شیب تند از ۲۰ تا ۵۰
        return round(20.0 + (dte - 3) * 15.0, 1)
    elif 5 <= dte < opt_min:
        # بین ۵ تا ۱۰ روز: رشد تدریجی از ۵۰ تا ۹۵
        return round(50.0 + (dte - 5) * ((95.0 - 50.0) / (opt_min - 5)), 1)
    elif opt_min <= dte <= opt_max:
        # بازه بهینه: ۱۰۰ کامل
        return 100.0
    elif opt_max < dte <= 60:
        # بین ۴۰ تا ۶۰ روز: افت ملایم از ۹۵ تا ۸۰
        return round(95.0 - (dte - opt_max) * (15.0 / 20.0), 1)
    elif 60 < dte <= 90:
        # بین ۶۰ تا ۹۰ روز: افت از ۸۰ تا ۵۵
        return round(80.0 - (dte - 60) * (25.0 / 30.0), 1)
    else:
        # بیش از ۹۰ روز: امتیاز ثابت ۴۵
        return 45.0


def calculate_underlying_readiness_score(
    row: pd.Series,
    is_call: bool,
    config: AppConfig,
) -> Tuple[float, List[str], Dict[str, float]]:
    """
    محاسبه امتیاز آمادگی دارایی پایه با منطق نامتقارن ۴ مؤلفه‌ای:
    هر مؤلفه مستقلاً امتیاز ۰ تا ۱۰۰ می‌گیرد و میانگین وزنی مساوی ۲۵٪ دارند:
    ۱. روند میانگین‌ها (SMA5 و SMA20)
    ۲. شاخص RSI-14
    ۳. صف و عمق بوک در سهم پایه
    ۴. پولبک و مومنتوم
    سپس اعمال سقف ضد-Chasing با فالبک هوشمند ۱ روزه (آستانه ۵٪).
    """
    reasons = []

    is_pullback = bool(row.get("پولبک پایه", False))
    dist_sma5 = float(row.get("فاصله پایه از SMA5 (%)", 0.0))
    dist_sma20 = float(row.get("فاصله پایه از SMA20 (%)", 0.0))
    queue = str(row.get("وضعیت صف پایه", "متعادل"))
    rsi = float(row.get("RSI14 پایه", 50.0))
    rsi_status = str(row.get("وضعیت RSI پایه", "عادی"))
    ret_1d = float(row.get("بازدهی ۱ روزه پایه (%)", 0.0))
    ret_3d = float(row.get("بازدهی ۳ روزه پایه (%)", 0.0))
    ret_5d = float(row.get("بازدهی ۵ روزه پایه (%)", 0.0))

    score_sma = 0.0
    score_rsi = 0.0
    score_queue = 0.0
    score_pullback = 0.0

    if is_call:
        # ==================== ۱. قواعد دارایی پایه برای Call ====================
        # ۱.۱ روند میانگین‌ها (۰ تا ۱۰۰): تثبیت بالای SMA5 + تقاطع صعودی SMA20
        if dist_sma5 >= 0 and dist_sma20 >= 0:
            score_sma = 100.0
            reasons.append("قیمت تثبیت‌شده بالای SMA5 و SMA20")
        elif dist_sma5 >= 0:
            score_sma = 80.0
            reasons.append("قیمت بالای میانگین کوتاه‌مدت SMA5")
        elif abs(dist_sma20) <= 2.5:
            score_sma = 65.0
            reasons.append("قیمت روی حمایت معتبر SMA20")
        elif dist_sma5 >= -2.0:
            score_sma = 40.0
        else:
            score_sma = 20.0

        # ۱.۲ شاخص RSI-14 (۰ تا ۱۰۰): برگشت از ۳۰ یا خروج از اشباع فروش
        if "خروج از اشباع فروش" in rsi_status or (30.0 <= rsi <= 45.0 and ret_1d > 0):
            score_rsi = 100.0
            reasons.append(f"برگشت RSI از اشباع فروش ({rsi:.1f})")
        elif 45.0 < rsi <= 65.0:
            score_rsi = 85.0
            reasons.append(f"شاخص RSI در منطقه صعودی متعادل ({rsi:.1f})")
        elif rsi > 70.0:
            score_rsi = 35.0  # احتیاط اشباع خرید
            reasons.append(f"RSI در محدوده اشباع خرید ({rsi:.1f})")
        else:
            score_rsi = 50.0

        # ۱.۳ دفتر سفارش و صف (۰ تا ۱۰۰): برتری خریدار / صف خرید
        if "صف خرید" in queue:
            score_queue = 100.0
            reasons.append("صف خرید در دارایی پایه")
        elif "تقاضا" in queue:
            score_queue = 75.0
            reasons.append("برتری محسوس تقاضا در بوک سفارشات پایه")
        elif "متعادل" in queue:
            score_queue = 50.0
        else:
            score_queue = 15.0

        # ۱.۴ پولبک (۰ تا ۱۰۰): پولبک به حمایت و برگشت تقاضا
        if is_pullback:
            score_pullback = 100.0
            reasons.append("پولبک فنی کم‌حجم به حمایت و برگشت تقاضا")
        elif ret_5d > 2.0 and ret_1d > 0:
            score_pullback = 80.0
            reasons.append("روند صعودی پرقدرت ۵ روزه")
        elif ret_1d > 0.5:
            score_pullback = 60.0
        else:
            score_pullback = 25.0

    else:
        # ==================== ۲. قواعد دارایی پایه برای Put ====================
        # ۲.۱ روند میانگین‌ها (۰ تا ۱۰۰): شکست زیر SMA5 + تقاطع نزولی SMA20
        if dist_sma5 < 0 and dist_sma20 < 0:
            score_sma = 100.0
            reasons.append("شکست نزولی به زیر SMA5 و SMA20")
        elif dist_sma5 < 0:
            score_sma = 80.0
            reasons.append("ریزش قیمت زیر SMA5")
        elif dist_sma20 < 0:
            score_sma = 65.0
            reasons.append("قرارگیری زیر باند SMA20")
        else:
            score_sma = 25.0

        # ۲.۲ شاخص RSI-14 (۰ تا ۱۰۰): شکست از ۷۰ یا ریزش از اشباع خرید
        if rsi >= 70.0 or (rsi >= 60.0 and ret_1d < -0.5):
            score_rsi = 100.0
            reasons.append(f"ریزش از اشباع خرید یا واگرایی منفی RSI ({rsi:.1f})")
        elif 35.0 <= rsi < 55.0:
            score_rsi = 80.0
            reasons.append(f"شاخص RSI در فاز نزولی پایدار ({rsi:.1f})")
        elif rsi < 30.0:
            score_rsi = 35.0  # احتیاط اشباع فروش
            reasons.append(f"RSI در اشباع فروش سهم ({rsi:.1f})")
        else:
            score_rsi = 50.0

        # ۲.۳ دفتر سفارش و صف (۰ تا ۱۰۰): برتری فروشنده / صف فروش
        if "صف فروش" in queue:
            score_queue = 100.0
            reasons.append("صف فروش قفل‌شده در دارایی پایه")
        elif "عرضه" in queue:
            score_queue = 75.0
            reasons.append("برتری محسوس عرضه در سفارشات پایه")
        elif "متعادل" in queue:
            score_queue = 50.0
        else:
            score_queue = 15.0

        # ۲.۴ پولبک (۰ تا ۱۰۰): پولبک به مقاومت و پس‌زده‌شدن
        if ret_5d < -2.0 and ret_1d < 0:
            score_pullback = 100.0
            reasons.append("موج نزولی قوی هم‌سو با Put")
        elif dist_sma5 < -1.5:
            score_pullback = 80.0
            reasons.append("پولبک به مقاومت SMA و ریجکت قیمت")
        elif ret_1d < -0.5:
            score_pullback = 60.0
        else:
            score_pullback = 25.0

    # میانگین وزنی مساوی ۲۵٪ از ۴ مؤلفه
    raw_readiness = (score_sma * 0.25) + (score_rsi * 0.25) + (score_queue * 0.25) + (score_pullback * 0.25)

    # ==================== ۳. سقف ضد-Chasing دینامیک و فالبک ۱‌روزه ====================
    anti_cfg = config.unified_scoring.anti_chasing if config else None
    penalty_factor = getattr(anti_cfg, "penalty_factor", 0.70) if anti_cfg else 0.70

    u_symbol = str(row.get("دارایی پایه", row.get("Symbol", "")))
    p_band = row.get("دامنه نوسان پایه", row.get("PriceBand"))

    th_3d, th_1d, band_used = get_chasing_thresholds(
        symbol=u_symbol,
        price_band=p_band,
        config=config,
    )

    has_3d_data = pd.notna(ret_3d) and abs(ret_3d) > 0.001
    has_chasing = False

    if has_3d_data:
        if is_call and ret_3d > th_3d:
            has_chasing = True
            reasons.append(f"اعمال جریمه ضد-Chasing (رشد ۳ روزه {ret_3d:+.1f}% فراتر از {th_3d:.1f}%)")
        elif not is_call and ret_3d < -th_3d:
            has_chasing = True
            reasons.append(f"اعمال جریمه ضد-Chasing (ریزش ۳ روزه {ret_3d:+.1f}% فراتر از -{th_3d:.1f}%)")
    else:
        # فالبک هوشمند به بازدهی ۱ روزه متناسب با سقف واقعی نوسان روزانه (عادی ۲.۷٪، اهرمی ۳.۶٪)
        if is_call and ret_1d > th_1d:
            has_chasing = True
            reasons.append(f"اعمال جریمه ضد-Chasing (فالبک بازده ۱ روزه {ret_1d:+.1f}% فراتر از {th_1d:.1f}%)")
        elif not is_call and ret_1d < -th_1d:
            has_chasing = True
            reasons.append(f"اعمال جریمه ضد-Chasing (فالبک بازده ۱ روزه {ret_1d:+.1f}% فراتر از -{th_1d:.1f}%)")

    if has_chasing:
        final_readiness = round(raw_readiness * penalty_factor, 1)
    else:
        final_readiness = round(min(raw_readiness, 100.0), 1)

    sub_components = {
        "sma": score_sma,
        "rsi": score_rsi,
        "queue": score_queue,
        "pullback": score_pullback,
        "raw": raw_readiness,
        "anti_chasing_applied": has_chasing,
        "threshold_3d": th_3d,
        "threshold_1d": th_1d,
        "daily_band": band_used,
    }

    return final_readiness, reasons, sub_components


def get_chasing_thresholds(
    symbol: str = "",
    price_band: Optional[float] = None,
    config: Optional[AppConfig] = None,
) -> Tuple[float, float, float]:
    """
    محاسبه آستانه‌های پویا و دقیق ضد-Chasing بر اساس سقف نوسان واقعی بورس تهران:
    - ورودی: نماد پایه، دامنه نوسان استخراج‌شده از TSETMC (در صورت وجود)، کانفیگ سیستم.
    - فرمول مصوب شورا:
        band = PriceBand (یا fallback: 0.04 اهرمی / 0.03 عادی)
        max_3day_move = ((1 + band) ** 3) - 1
        threshold_3day = 0.90 * max_3day_move * 100   # عادی ≈ 8.35% | اهرمی ≈ 11.24%
        threshold_1day = 0.90 * band * 100           # عادی ≈ 2.70% | اهرمی ≈ 3.60%
    - خروجی: (آستانه ۳‌روزه به درصد، آستانه ۱‌روزه به درصد، دامنه روزانه به اعشار)
    """
    anti_cfg = config.unified_scoring.anti_chasing if config else None
    safety_ratio = getattr(anti_cfg, "chasing_safety_ratio", 0.90) if anti_cfg else 0.90
    daily_bands = getattr(anti_cfg, "daily_bands", {"leveraged": 0.04, "regular": 0.03}) if anti_cfg else {"leveraged": 0.04, "regular": 0.03}
    lev_symbols = getattr(anti_cfg, "leveraged_symbols", ["اهرم", "توان", "موج", "اطلس", "جهش", "شتاب", "نارنج", "بیدار"]) if anti_cfg else ["اهرم", "توان", "موج", "اطلس", "جهش", "شتاب", "نارنج", "بیدار"]

    # ۱. اولویت اول: استفاده از دامنه نوسان پویای استخراج‌شده از TSETMC
    band = None
    if price_band is not None and pd.notna(price_band) and float(price_band) > 0.005:
        band = float(price_band)
    else:
        # ۲. اولویت دوم: فالبک بر اساس نوع نماد (اهرمی یا عادی)
        norm_s = normalize_fa(symbol) if symbol else ""
        is_lev = any(symbols_match(norm_s, s) for s in lev_symbols)
        band = float(daily_bands.get("leveraged" if is_lev else "regular", 0.04 if is_lev else 0.03))

    max_3day_move = ((1.0 + band) ** 3) - 1.0
    threshold_3day = round(safety_ratio * max_3day_move * 100.0, 2)
    threshold_1day = round(safety_ratio * band * 100.0, 2)
    return threshold_3day, threshold_1day, band


def calculate_combined_liquidity_score(
    structural_pct: float,
    today_volume: float,
    avg_5d_volume: float,
    config: AppConfig,
) -> Tuple[float, float, float]:
    """
    محاسبه امتیاز نقدینگی ترکیبی (۳۰٪ کل):
    - نقدینگی ساختاری (۱۸٪ سهم کل): رتبه صدکی میانگین ۵ روزه ارزش و تعداد معاملات
    - جهش لحظه‌ای (۱۲٪ سهم کل): فرمول مصوب شورا:
        momentum_ratio = today_volume / max(avg_5d_volume, 1)
        liquidity_spike_score = min(momentum_ratio / 3.0, 1.0) * 100
        (نسبت ۳ برابر یا بیشتر معادل امتیاز کامل ۱۰۰)
    خروجی: (امتیاز کل نقدینگی از ۱۰۰, امتیاز ساختاری از ۱۰۰, امتیاز جهش از ۱۰۰)
    """
    liq_cfg = config.unified_scoring.liquidity_breakdown
    w_struct = liq_cfg.structural_weight  # 0.18
    w_spike = liq_cfg.spike_weight       # 0.12
    total_w = w_struct + w_spike         # 0.30

    structural_score = float(structural_pct)

    # فرمول دقیق و سقف‌دار جهش لحظه‌ای
    safe_avg = max(float(avg_5d_volume), 1.0)
    momentum_ratio = float(today_volume) / safe_avg
    spike_score = round(min(momentum_ratio / 3.0, 1.0) * 100.0, 1)

    combined_score = round(
        (structural_score * w_struct + spike_score * w_spike) / total_w, 1
    )
    return combined_score, structural_score, spike_score


def calculate_score(
    row: pd.Series,
    structural_liq_pct: float,
    bubble_pct_rank: Optional[float],
    config: AppConfig,
) -> Dict[str, Any]:
    """
    تابع واحد و ایمن امتیازدهی اسکنر آپشن (Calculate Score)
    ورودی: ردیف قرارداد، رتبه صدکی ساختاری ۵ روزه، رتبه صدکی حباب، و کانفیگ سیستم.
    خروجی: دیکشنری شامل امتیاز نهایی (۰ تا ۱۰۰)، ۴ زیرامتیاز، دلایل و وضعیت فیلتر سخت تفکیک‌شده Call/Put.
    ایمن در برابر مقادیر NaN یا داده خراب.
    """
    opt_type_raw = str(row.get("نوع قرارداد", "")).lower()
    is_call = "call" in opt_type_raw or "خرید" in opt_type_raw or str(row.get("نماد", "")).startswith("ض")

    # ۱. آمادگی دارایی پایه (۲۵٪)
    readiness_score, reasons_readiness, readiness_detail = calculate_underlying_readiness_score(
        row, is_call, config
    )

    # ۲. نقدینگی ترکیبی (۳۰٪) با فرمول مصوب شورا
    val_today = float(row.get("ارزش معاملات امروز (ریال)", 0.0))
    avg_5d = float(row.get("میانگین ارزش ۵ روزه (ریال)", 0.0))
    comb_liq_score, struct_score, spike_score = calculate_combined_liquidity_score(
        structural_pct=structural_liq_pct,
        today_volume=val_today,
        avg_5d_volume=avg_5d,
        config=config,
    )

    # ۳. ارزش نسبی / حباب BSM (۳۰٪) — ایمن‌سازی کامل خطای NaN و Deep OTM
    is_deep_otm = bool(row.get("عمیقاً بی‌ارزش", False))

    # بررسی مقادیر BSM و دلتا: در صورت نامعتبر، NaN، None بودن یا زیر آستانه، قرارداد Deep OTM محسوب می‌شود
    if "قیمت تئوریک BSM" in row:
        bsm_val = row.get("قیمت تئوریک BSM")
        if bsm_val is None or pd.isna(bsm_val):
            is_deep_otm = True
        else:
            try:
                if float(bsm_val) < 10.0:
                    is_deep_otm = True
            except (ValueError, TypeError):
                is_deep_otm = True

    if "دلتا" in row:
        delta_val = row.get("دلتا")
        if delta_val is None or pd.isna(delta_val):
            is_deep_otm = True
        else:
            try:
                if abs(float(delta_val)) < 0.10:
                    is_deep_otm = True
            except (ValueError, TypeError):
                is_deep_otm = True

    if is_deep_otm:
        rel_val_score = 0.0
    elif bubble_pct_rank is None or np.isnan(bubble_pct_rank):
        rel_val_score = 50.0
    else:
        rel_val_score = round(max(0.0, min(100.0, 100.0 - float(bubble_pct_rank))), 1)

    # ۴. تناسب DTE (۱۵٪)
    dte = float(row.get("روزهای تا سررسید (DTE)", 0.0))
    dte_score = calculate_dte_suitability_score(
        dte=dte,
        opt_min=config.unified_scoring.dte_curve.optimal_min,
        opt_max=config.unified_scoring.dte_curve.optimal_max,
    )

    # ۵. ترکیب با وزن‌های نهایی (مجموع ۱۰۰٪)
    w = config.unified_scoring.weights
    final_score = round(
        readiness_score * w.underlying_readiness
        + comb_liq_score * w.combined_liquidity
        + rel_val_score * w.relative_value
        + dte_score * w.dte_suitability,
        1,
    )

    # ۶. بررسی واجد شرایط بودن فیلتر سخت تفکیک‌شده Call و Put (Hard Exclusion)
    hard = config.unified_scoring.hard_filter
    trade_count = int(row.get("تعداد معاملات امروز", 0))

    if is_call:
        min_val = getattr(getattr(hard, "call", None), "min_trade_value_rials", hard.min_trade_value_rials)
        min_trades = getattr(getattr(hard, "call", None), "min_trade_count", hard.min_trade_count)
    else:
        min_val = getattr(getattr(hard, "put", None), "min_trade_value_rials", 1_000_000_000.0)
        min_trades = getattr(getattr(hard, "put", None), "min_trade_count", 10)

    passes_hard_filter = bool(
        val_today >= min_val
        and trade_count >= min_trades
        and dte >= hard.min_dte
        and not is_deep_otm
    )

    # دلایل کلی
    reasons_all = list(reasons_readiness)
    val_tomans = val_today / 10_000_000.0
    if val_tomans >= 1000:
        val_str = f"{round(val_tomans / 1000, 1)} میلیارد تومان"
    else:
        val_str = f"{round(val_tomans)} میلیون تومان"

    if is_deep_otm:
        reasons_all.insert(0, "اختیار عمیقاً بی‌ارزش (خارج از پیشنهادهای ورود)")
    elif spike_score >= 80:
        reasons_all.insert(0, f"جهش نقدینگی لحظه‌ای مطلوب ({val_str})")
    elif comb_liq_score >= 70:
        reasons_all.insert(0, f"نقدینگی و عمق ساختاری بالا ({val_str})")
    else:
        reasons_all.insert(0, f"ارزش معاملات {val_str}")

    if rel_val_score >= 80:
        reasons_all.append("قیمت‌گذاری منصفانه نسبت به ارزش تئوریک BSM")
    elif rel_val_score >= 50:
        reasons_all.append("حباب قیمتی در دامنه معقول بازار")

    if dte_score >= 95:
        reasons_all.append(f"بازه سررسید بهینه ({int(dte)} روز)")
    else:
        reasons_all.append(f"سررسید {int(dte)} روزه")

    explanation_text = "؛ ".join(reasons_all[:4]) + "."

    return {
        "final_score": final_score,
        "readiness_score": readiness_score,
        "combined_liquidity_score": comb_liq_score,
        "structural_liquidity_score": struct_score,
        "spike_liquidity_score": spike_score,
        "relative_value_score": rel_val_score,
        "dte_suitability_score": dte_score,
        "passes_hard_filter": passes_hard_filter,
        "explanation": explanation_text,
        "reasons": reasons_all,
        "is_deep_otm": is_deep_otm,
    }


def select_top_n_diversified(
    scored_contracts: Any,
    n: int = 10,
    max_per_underlying: int = 3,
) -> Any:
    """
    انتخاب n قرارداد برتر با اعمال سقف تنوع در دارایی پایه (حداکثر max_per_underlying از هر دارایی پایه).
    پشتیبانی هم از pandas.DataFrame و هم از لیست قراردادها (اشیاء یا دیکشنری).
    """
    if isinstance(scored_contracts, pd.DataFrame):
        if scored_contracts.empty:
            return scored_contracts.copy()

        df_sorted = scored_contracts.sort_values("امتیاز الگوریتم", ascending=False).copy()

        # محاسبه رتبه درون دارایی پایه در صورت عدم وجود
        if "رتبه در دارایی پایه" not in df_sorted.columns and "دارایی پایه" in df_sorted.columns:
            df_sorted["رتبه در دارایی پایه"] = (
                df_sorted.groupby("دارایی پایه")["امتیاز الگوریتم"]
                .rank(ascending=False, method="first")
                .astype(int)
            )

        selected_indices = []
        counts: Dict[str, int] = {}

        for idx, row in df_sorted.iterrows():
            u = str(row.get("دارایی پایه", ""))
            current_count = counts.get(u, 0)
            if current_count < max_per_underlying:
                selected_indices.append(idx)
                counts[u] = current_count + 1
                if len(selected_indices) == n:
                    break

        return df_sorted.loc[selected_indices].reset_index(drop=True)

    elif isinstance(scored_contracts, list):
        def get_score(x):
            if hasattr(x, "score"):
                return getattr(x, "score", 0.0)
            elif isinstance(x, dict):
                return x.get("امتیاز الگوریتم", x.get("score", 0.0))
            return 0.0

        def get_underlying(x):
            if hasattr(x, "underlying"):
                return getattr(x, "underlying", "")
            elif isinstance(x, dict):
                return x.get("دارایی پایه", x.get("underlying", ""))
            return ""

        sorted_list = sorted(scored_contracts, key=get_score, reverse=True)
        selected = []
        count_per_underlying: Dict[str, int] = {}
        for contract in sorted_list:
            u = get_underlying(contract)
            if count_per_underlying.get(u, 0) < max_per_underlying:
                selected.append(contract)
                count_per_underlying[u] = count_per_underlying.get(u, 0) + 1
                if len(selected) == n:
                    break
        return selected

def get_sister_contracts(
    current_symbol: str,
    underlying: str,
    df_pool: pd.DataFrame,
    is_call: Optional[bool] = None,
    max_sisters: int = 2,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    استخراج ۱ یا حداکثر ۲ نماد جایگزین همنام (Sister Contracts) معتبر و واجد فیلترهای سخت
    روی همان دارایی پایه و با همان نوع (Call یا Put)، به جز خود نماد جاری، با بالاترین امتیاز نهایی.
    """
    if df_pool is None or df_pool.empty:
        return []

    # انعطاف‌پذیری در نام آرگومان‌ها
    current_symbol = kwargs.get("symbol", current_symbol)
    if "df_eligible" in kwargs and kwargs["df_eligible"] is not None:
        df_pool = kwargs["df_eligible"]

    df = df_pool.copy()

    # شرط فقط واجدین شرایط فیلتر سخت (در صورت وجود ستون)
    if "واجد فیلتر سخت" in df.columns:
        df = df[df["واجد فیلتر سخت"] == True]

    # حذف قراردادهای عمیقاً بی‌ارزش
    if "عمیقاً بی‌ارزش" in df.columns:
        df = df[df["عمیقاً بی‌ارزش"] == False]

    # همپایه بودن و غیر از نماد جاری
    mask = (df["دارایی پایه"] == underlying) & (df["نماد"] != current_symbol)

    # تفکیک بر اساس نوع قرارداد (Call یا Put)
    if is_call is not None:
        call_mask = df["نوع قرارداد"].str.contains("Call|خرید", case=False, na=False) | df["نماد"].str.startswith("ض")
        if is_call:
            mask = mask & call_mask
        else:
            mask = mask & (~call_mask)

    df_filtered = df[mask]
    if df_filtered.empty:
        return []

    score_col = "امتیاز الگوریتم" if "امتیاز الگوریتم" in df_filtered.columns else "final_score"
    sisters_df = df_filtered.sort_values(score_col, ascending=False).head(max_sisters)

    results = []
    for _, row in sisters_df.iterrows():
        results.append({
            "symbol": str(row["نماد"]),
            "strike": int(row.get("قیمت اعمال", 0)),
            "dte": int(row.get("روزهای تا سررسید (DTE)", 0)),
            "score": float(row.get(score_col, 0.0)),
            "price": int(row.get("قیمت پایانی بازار", 0)),
            "underlying": underlying,
        })
    return results


def compute_top_call_and_put(
    df_all: pd.DataFrame,
    config: AppConfig,
    top_n: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    استخراج هم‌زمان ۱۰ پیشنهاد برتر خرید (Call) و ۱۰ پیشنهاد برتر فروش (Put)
    با آستانه تفکیک‌شده Call (۵۰۰ م.ت و ۳۰ معامله) و Put (۱۰۰ م.ت و ۱۰ معامله)،
    سقف تنوع دارایی پایه (حداکثر ۳ از هر نماد)، مهار Deep OTM و بررسی هشدار تمرکز.
    """
    if df_all.empty:
        empty_res = {"call_warning": False, "put_warning": False, "call_sym": None, "put_sym": None}
        return pd.DataFrame(), pd.DataFrame(), empty_res

    logger.info("در حال استخراج لیست‌های ۱۰تایی برتر Call و Put بر اساس نسخه نهایی پچ v2...")

    # محاسبه رتبه صدکی نقدینگی ساختاری ۵ روزه و حباب برای کل جدول
    n = len(df_all)
    avg_5d_ranks = rankdata(df_all["میانگین ارزش ۵ روزه (ریال)"], method="average")
    trade_ranks = rankdata(df_all["تعداد معاملات امروز"], method="average")
    structural_pcts = ((avg_5d_ranks / n) * 0.70 + (trade_ranks / n) * 0.30) * 100.0

    valid_b_mask = df_all["حباب خام (%)"].notna() & (~df_all.get("عمیقاً بی‌ارزش", False))
    n_valid = int(valid_b_mask.sum())
    bubble_pct_series = pd.Series(index=df_all.index, dtype=float)
    if n_valid > 1:
        b_ranks = rankdata(df_all.loc[valid_b_mask, "حباب خام (%)"], method="average")
        bubble_pct_series.loc[valid_b_mask] = (b_ranks / n_valid) * 100.0
    elif n_valid == 1:
        bubble_pct_series.loc[valid_b_mask] = 50.0

    # ارزیابی تک‌تک قراردادها با calculate_score
    scores = []
    readiness_list = []
    liq_list = []
    struct_liq_list = []
    spike_liq_list = []
    rel_val_list = []
    dte_list = []
    pass_hard_list = []
    explanations = []

    for idx, (_, row) in enumerate(df_all.iterrows()):
        res = calculate_score(
            row=row,
            structural_liq_pct=structural_pcts[idx],
            bubble_pct_rank=bubble_pct_series.iloc[idx],
            config=config,
        )
        scores.append(res["final_score"])
        readiness_list.append(res["readiness_score"])
        liq_list.append(res["combined_liquidity_score"])
        struct_liq_list.append(res["structural_liquidity_score"])
        spike_liq_list.append(res["spike_liquidity_score"])
        rel_val_list.append(res["relative_value_score"])
        dte_list.append(res["dte_suitability_score"])
        pass_hard_list.append(res["passes_hard_filter"])
        explanations.append(res["explanation"])

    df_scored = df_all.copy()
    df_scored["امتیاز الگوریتم"] = scores
    df_scored["امتیاز آمادگی پایه"] = readiness_list
    df_scored["امتیاز نقدینگی ترکیبی"] = liq_list
    df_scored["امتیاز نقدینگی ساختاری"] = struct_liq_list
    df_scored["امتیاز جهش لحظه‌ای"] = spike_liq_list
    df_scored["امتیاز ارزش نسبی (حباب)"] = rel_val_list
    df_scored["امتیاز تناسب DTE"] = dte_list
    df_scored["واجد فیلتر سخت"] = pass_hard_list
    df_scored["چرا این امتیاز"] = explanations

    # فیلتر سخت: فقط قراردادهای واجد شرایط وارد لیست‌های برتر می‌شوند
    df_eligible = df_scored[df_scored["واجد فیلتر سخت"]].copy()

    # تفکیک Call و Put
    is_call_series = df_eligible["نوع قرارداد"].str.contains("Call|خرید", case=False, na=False) | df_eligible["نماد"].str.startswith("ض")

    df_calls_pool = df_eligible[is_call_series].sort_values("امتیاز الگوریتم", ascending=False).copy()
    df_puts_pool = df_eligible[~is_call_series].sort_values("امتیاز الگوریتم", ascending=False).copy()

    # محاسبه رتبه درون دارایی پایه برای تمام قراردادهای واجد شرایط Call و Put
    if not df_calls_pool.empty:
        df_calls_pool["رتبه در دارایی پایه"] = (
            df_calls_pool.groupby("دارایی پایه")["امتیاز الگوریتم"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
    else:
        df_calls_pool["رتبه در دارایی پایه"] = pd.Series(dtype=int)

    if not df_puts_pool.empty:
        df_puts_pool["رتبه در دارایی پایه"] = (
            df_puts_pool.groupby("دارایی پایه")["امتیاز الگوریتم"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
    else:
        df_puts_pool["رتبه در دارایی پایه"] = pd.Series(dtype=int)

    # استخراج برترین‌ها با اعمال سقف تنوع دارایی پایه (حداکثر max_per_underlying)
    max_per_u = getattr(config.unified_scoring, "max_per_underlying", 3)
    df_calls = select_top_n_diversified(df_calls_pool, n=top_n, max_per_underlying=max_per_u)
    df_puts = select_top_n_diversified(df_puts_pool, n=top_n, max_per_underlying=max_per_u)

    # استخراج نمادهای جایگزین همنام (Sister Contracts) برای هر کارت برتر
    if not df_calls.empty:
        sisters_calls = []
        for _, r in df_calls.iterrows():
            s = get_sister_contracts(
                current_symbol=r["نماد"],
                underlying=r["دارایی پایه"],
                df_pool=df_calls_pool,
                is_call=True,
                max_sisters=2,
            )
            sisters_calls.append(s)
        df_calls["قراردادهای جایگزین"] = sisters_calls
    else:
        df_calls["قراردادهای جایگزین"] = pd.Series(dtype=object)

    if not df_puts.empty:
        sisters_puts = []
        for _, r in df_puts.iterrows():
            s = get_sister_contracts(
                current_symbol=r["نماد"],
                underlying=r["دارایی پایه"],
                df_pool=df_puts_pool,
                is_call=False,
                max_sisters=2,
            )
            sisters_puts.append(s)
        df_puts["قراردادهای جایگزین"] = sisters_puts
    else:
        df_puts["قراردادهای جایگزین"] = pd.Series(dtype=object)

    # بررسی هشدار تمرکز نماد پایه (> max_same نماد در هر لیست)
    max_same = config.unified_scoring.max_same_underlying_alert

    conc_info = {
        "call_warning": False,
        "call_sym": None,
        "call_count": 0,
        "put_warning": False,
        "put_sym": None,
        "put_count": 0,
    }

    if not df_calls.empty:
        c_counts = df_calls["دارایی پایه"].value_counts()
        for sym, cnt in c_counts.items():
            if cnt > max_same:
                conc_info["call_warning"] = True
                conc_info["call_sym"] = sym
                conc_info["call_count"] = int(cnt)
                break

    if not df_puts.empty:
        p_counts = df_puts["دارایی پایه"].value_counts()
        for sym, cnt in p_counts.items():
            if cnt > max_same:
                conc_info["put_warning"] = True
                conc_info["put_sym"] = sym
                conc_info["put_count"] = int(cnt)
                break

    logger.info(
        f"استخراج با موفقیت انجام شد: {len(df_calls)} اختیار خرید و {len(df_puts)} اختیار فروش برتر با سقف تنوع انتخاب شدند."
    )
    return df_calls, df_puts, conc_info
