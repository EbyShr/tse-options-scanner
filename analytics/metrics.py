"""
ماژول پردازش جامع متریک‌ها، محاسبه حباب، اهرم، نقدینگی، امتیاز ترکیبی و رتبه‌بندی صدکی
"""

import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config import AppConfig
from data.normalizer import normalize_fa
from analytics.greeks_bsm import calculate_bsm_price, calculate_leverage, evaluate_moneyness
from analytics.underlying_momentum import determine_momentum_label

logger = logging.getLogger("OptionScanner.Metrics")


def process_options_dataframe(
    df_options: pd.DataFrame,
    underlying_stats: Dict[str, Dict[str, Any]],
    options_avg_values: Dict[str, float],
    config: AppConfig,
) -> pd.DataFrame:
    """
    محاسبه تمام متریک‌های درخواستی برای تک‌تک قراردادها، رتبه‌بندی صدکی و محاسبه امتیاز نهایی ترکیبی
    """
    if df_options.empty:
        return pd.DataFrame()

    logger.info("در حال محاسبه متریک‌های ارزش‌گذاری، نقدینگی، اهرم و حباب...")
    rows = []

    for _, row in df_options.iterrows():
        sym = str(row.get("Symbol", "")).strip()
        inscode = str(row.get("InsCode", "")).strip()
        raw_underlying = str(row.get("UnderlyingSymbol", "")).strip()
        norm_underlying = normalize_fa(raw_underlying)

        # دریافت آمار مربوط به دارایی پایه
        u_stats = underlying_stats.get(raw_underlying) or underlying_stats.get(norm_underlying) or {}
        spot = float(u_stats.get("Close", 0.0))
        if spot <= 0:
            spot = float(row.get("Spot", row.get("UnderlyingClose", row.get("UnderlyingLast", 0.0))))

        strike = float(row.get("Strike", 0.0))
        dte = float(row.get("DaysToExpiry", 0.0))

        # قیمت بازاری آپشن (اولویت با Close، سپس Last)
        close_price = float(row.get("Close", 0.0))
        last_price = float(row.get("Last", 0.0))
        market_price = close_price if close_price > 0 else last_price

        # نوع قرارداد (Call / Put)
        opt_type_raw = str(row.get("OptionType", "")).lower()
        is_call = "call" in opt_type_raw or "خرید" in opt_type_raw or sym.startswith("ض")
        opt_type_fa = "اختیار خرید (Call)" if is_call else "اختیار فروش (Put)"

        # نوسان‌پذیری (IV استخراج‌شده یا Realized Volatility دارایی پایه)
        iv = row.get("ImpliedVolatility")
        realized_vol = float(u_stats.get("RealizedVol", 0.35))
        if pd.notna(iv) and float(iv) > 0.02 and float(iv) < 2.5:
            vol_used = float(iv)
            vol_source = "IV بازار"
        else:
            vol_used = realized_vol
            vol_source = "نوسان تاریخی پایه"

        # ۱. محاسبه قیمت تئوریک BSM و دلتا
        bsm_price, delta = calculate_bsm_price(
            spot=spot,
            strike=strike,
            dte=dte,
            rate=config.valuation.risk_free_rate,
            volatility=vol_used,
            option_type="call" if is_call else "put",
            dividend_yield=config.valuation.dividend_yield,
        )

        # دلتا پکیج در صورت وجود
        pkg_delta = row.get("Delta")
        if pd.notna(pkg_delta) and abs(float(pkg_delta)) <= 1.0:
            delta = float(pkg_delta)

        # ۲. بررسی Deep OTM، مهار انفجار حباب و ایمن‌سازی کامل خطای داده نامعتبر / NaN
        min_bsm_cfg = getattr(getattr(config, "unified_scoring", None), "deep_otm_cap", None)
        min_bsm_price = min_bsm_cfg.min_bsm_price if min_bsm_cfg else 10.0
        min_delta = min_bsm_cfg.min_delta if min_bsm_cfg else 0.10

        # اگر دلتا یا BSM نامعتبر، None، یا NaN باشد، حالت داده نامعتبر/Deep OTM در نظر گرفته می‌شود
        is_invalid_bsm = pd.isna(bsm_price) or bsm_price is None or bsm_price <= 0
        is_invalid_delta = pd.isna(delta) or delta is None

        if is_invalid_bsm or is_invalid_delta or (bsm_price < min_bsm_price) or (abs(delta) < min_delta):
            is_deep_otm = True
            bubble_score_raw = None
            deep_otm_label = "لاتاری/بی‌ارزش عمیق"
            bubble_rial = round(market_price - (bsm_price if (not is_invalid_bsm and bsm_price is not None) else 0.0))
        else:
            is_deep_otm = False
            bubble_score_raw = round(((market_price - bsm_price) / bsm_price) * 100.0, 2)
            bubble_rial = round(market_price - bsm_price)
            deep_otm_label = ""

        # ۳. محاسبه اهرم (Leverage)
        leverage = calculate_leverage(spot=spot, option_price=market_price, delta=delta)

        # ۴. نقدینگی واقعی (Liquidity)
        val_today = float(row.get("Value", 0.0))
        trade_count = int(row.get("TradeCount", 0))
        avg_val_5d = options_avg_values.get(inscode, 0.0)

        # نسبت ارزش معامله امروز به میانگین ۵ روزه
        if avg_val_5d > 0:
            val_to_avg5d_ratio = round(val_today / avg_val_5d, 2)
        else:
            val_to_avg5d_ratio = 1.0 if val_today > 0 else 0.0

        # پرچم نقدینگی (Liquidity Flag)
        is_illiquid = (
            val_today < config.liquidity_filter.min_trade_value_rials
            or trade_count < config.liquidity_filter.min_trade_count
        )
        liquidity_flag = (
            config.liquidity_filter.low_liquidity_label
            if is_illiquid
            else config.liquidity_filter.normal_liquidity_label
        )

        # اطلاعات عمق دفتر سفارشات (Order Book)
        bid_price = float(row.get("BidPrice", 0.0))
        ask_price = float(row.get("AskPrice", 0.0))
        bid_vol = float(row.get("BidVolume", 0.0))
        ask_vol = float(row.get("AskVolume", 0.0))
        spread_abs = max(0.0, ask_price - bid_price) if (ask_price > 0 and bid_price > 0) else 0.0

        # ۵. وضعیت سودآوری (Moneyness) و درصد فاصله تا سربه‌سر
        moneyness, dist_breakeven_pct = evaluate_moneyness(
            spot=spot,
            strike=strike,
            option_price=market_price,
            option_type="call" if is_call else "put",
        )

        # ۶. جهت و مومنتوم دارایی پایه
        momentum_label = determine_momentum_label(u_stats)

        # تشخیص هم‌جهت بودن استراتژی آپشن با روند پایه
        if is_call and "صعودی" in momentum_label:
            alignment = "هم‌جهت با روند صعودی پایه ✓"
        elif not is_call and "نزولی" in momentum_label:
            alignment = "هم‌جهت با روند نزولی پایه ✓"
        elif is_call and "نزولی" in momentum_label:
            alignment = "خلاف روند (ریسک بالا ⚠)"
        elif not is_call and "صعودی" in momentum_label:
            alignment = "خلاف روند (ریسک بالا ⚠)"
        else:
            alignment = "خنثی / بدون سوگیری قوی"

        # تاریخ سررسید جلالی و میلادی
        end_date = str(row.get("EndDate", "")).split("T")[0]

        # محاسبه درصد تغییر قیمت آپشن نسبت به روز قبل (مثبت/منفی امروز)
        yesterday_price = float(row.get("Yesterday", 0.0))
        if yesterday_price > 0 and close_price > 0:
            price_change_today_pct = round(((close_price - yesterday_price) / yesterday_price) * 100.0, 2)
        elif yesterday_price > 0 and last_price > 0:
            price_change_today_pct = round(((last_price - yesterday_price) / yesterday_price) * 100.0, 2)
        else:
            price_change_today_pct = 0.0

        # تشخیص رشد ناگهانی حجم بدون سابقه (فومو)
        is_fomo_spike = bool(val_to_avg5d_ratio >= 6.0 and val_today >= 500_000_000 and avg_val_5d > 0)
        fomo_flag = "جهش ناگهانی حجم (احتمال فومو)" if is_fomo_spike else "حجم متناسب با سابقه"

        rows.append({
            "نماد": sym,
            "نام قرارداد": str(row.get("Name", sym)),
            "دارایی پایه": raw_underlying,
            "نوع قرارداد": opt_type_fa,
            "قیمت اعمال": round(strike),
            "قیمت پایه": round(spot),
            "قیمت پایانی بازار": round(close_price),
            "قیمت آخرین معامله": round(last_price),
            "قیمت دیروز": round(yesterday_price),
            "درصد تغییر امروز (%)": price_change_today_pct,
            "قیمت تئوریک BSM": round(bsm_price),
            "حباب خام (%)": bubble_score_raw,
            "حباب ریالی": bubble_rial,
            "عمیقاً بی‌ارزش": is_deep_otm,
            "برچسب سفته‌بازی": deep_otm_label,
            "اهرم": leverage,
            "دلتا": round(delta, 3),
            "نوسان‌پذیری": round(vol_used * 100.0, 1),
            "منبع نوسان": vol_source,
            "روزهای تا سررسید (DTE)": int(dte),
            "تاریخ سررسید": end_date,
            "وضعیت سودآوری": moneyness,
            "فاصله تا سربه‌سر (%)": dist_breakeven_pct,
            "ارزش معاملات امروز (ریال)": round(val_today),
            "تعداد معاملات امروز": trade_count,
            "میانگین ارزش ۵ روزه (ریال)": round(avg_val_5d),
            "نسبت معامله به میانگین": val_to_avg5d_ratio,
            "وضعیت نقدینگی": liquidity_flag,
            "کم‌عمق": is_illiquid,
            "پرچم فومو": fomo_flag,
            "جهش فومو": is_fomo_spike,
            "بهترین مظنه خرید (Bid)": round(bid_price),
            "حجم مظنه خرید": round(bid_vol),
            "بهترین مظنه فروش (Ask)": round(ask_price),
            "حجم مظنه فروش": round(ask_vol),
            "اسپرد (اختلاف مظنه)": round(spread_abs),
            "روند دارایی پایه": momentum_label,
            "هم‌جهتی با روند": alignment,
            "بازدهی ۱ روزه پایه (%)": u_stats.get("Return1D", 0.0),
            "بازدهی ۳ روزه پایه (%)": u_stats.get("Return3D", 0.0),
            "بازدهی ۵ روزه پایه (%)": u_stats.get("Return5D", 0.0),
            "بازدهی ۱۰ روزه پایه (%)": u_stats.get("Return10D", 0.0),
            "فاصله پایه از SMA5 (%)": u_stats.get("DistSMA5", 0.0),
            "فاصله پایه از SMA20 (%)": u_stats.get("DistSMA20", 0.0),
            "RSI14 پایه": u_stats.get("RSI14", 50.0),
            "وضعیت RSI پایه": u_stats.get("RSIStatus", "عادی"),
            "پولبک پایه": u_stats.get("IsPullback", False),
            "شرح پولبک": u_stats.get("PullbackDesc", "ندارد"),
            "وضعیت صف پایه": u_stats.get("QueueStatus", "متعادل"),
            "دامنه نوسان پایه": u_stats.get("PriceBand", 0.03),
            "آستانه Chasing ۳ روزه": u_stats.get("Threshold3D", 8.35),
            "آستانه Chasing ۱ روزه": u_stats.get("Threshold1D", 2.70),
            "نوع نماد پایه": "صندوق اهرمی" if u_stats.get("IsLeveraged", False) else "عادی",
            "InsCode": inscode,
        })

    df_res = pd.DataFrame(rows)
    if df_res.empty:
        return df_res

    # =========================================================================
    # ۷. رتبه‌بندی صدکی (Percentile Ranking) و امتیاز ترکیبی شفاف
    # =========================================================================
    n_items = len(df_res)
    if n_items > 1:
        # رتبه‌بندی حباب: هرچه حباب کمتر باشد برای خریدار جذاب‌تر است (معکوس)
        # برای نمادهای عمیقاً بی‌ارزش (Deep OTM)، حباب درصد N/A است و جذابیت حباب ۰ در نظر گرفته می‌شود
        valid_b_mask = df_res["حباب خام (%)"].notna() & (~df_res["عمیقاً بی‌ارزش"])
        n_valid = int(valid_b_mask.sum())

        df_res["رتبه صدکی حباب"] = np.nan
        df_res["جذابیت حباب (معکوس)"] = 0.0

        if n_valid > 1:
            bubble_ranks = rankdata(df_res.loc[valid_b_mask, "حباب خام (%)"], method="average")
            df_res.loc[valid_b_mask, "رتبه صدکی حباب"] = np.round((bubble_ranks / n_valid) * 100.0, 1)
            df_res.loc[valid_b_mask, "جذابیت حباب (معکوس)"] = np.round(100.0 - df_res.loc[valid_b_mask, "رتبه صدکی حباب"], 1)
        elif n_valid == 1:
            df_res.loc[valid_b_mask, "رتبه صدکی حباب"] = 50.0
            df_res.loc[valid_b_mask, "جذابیت حباب (معکوس)"] = 50.0

        # رتبه‌بندی نقدینگی بر اساس ارزش معامله و تعداد معاملات
        val_ranks = rankdata(df_res["ارزش معاملات امروز (ریال)"], method="average")
        trade_ranks = rankdata(df_res["تعداد معاملات امروز"], method="average")
        combined_liq_rank = 0.7 * (val_ranks / n_items) + 0.3 * (trade_ranks / n_items)
        df_res["رتبه صدکی نقدینگی"] = np.round(combined_liq_rank * 100.0, 1)

        # رتبه‌بندی اهرم (جزء مستقل ۱۵٪ - نسخه v4):
        # ترتیب فیلترهای سه‌گانه قبل از رتبه‌بندی نهایی اهرم:
        # مرحله ۱: گیت Deep-OTM (بدون تغییر)
        # مرحله ۲: گیت نقدینگی سخت (تفکیک Call و Put، ارزش و تعداد معاملات و DTE)
        # مرحله ۳: گیت سخت اهرم (حداقل ۳.۰، تفکیک Call و Put)
        hard = config.unified_scoring.hard_filter
        min_dte = getattr(hard, "min_dte", 3)
        min_val_call = getattr(getattr(hard, "call", None), "min_trade_value_rials", getattr(hard, "min_trade_value_rials", 5_000_000_000.0))
        min_trades_call = getattr(getattr(hard, "call", None), "min_trade_count", getattr(hard, "min_trade_count", 30))
        min_lev_call = getattr(getattr(hard, "call", None), "min_leverage", getattr(hard, "min_leverage_call", 3.0))

        min_val_put = getattr(getattr(hard, "put", None), "min_trade_value_rials", 1_000_000_000.0)
        min_trades_put = getattr(getattr(hard, "put", None), "min_trade_count", 10)
        min_lev_put = getattr(getattr(hard, "put", None), "min_leverage", getattr(hard, "min_leverage_put", 3.0))

        is_call_res = df_res["نوع قرارداد"].astype(str).str.contains("Call|خرید", case=False, na=False) | df_res["نماد"].astype(str).str.startswith("ض")
        req_val = np.where(is_call_res, min_val_call, min_val_put)
        req_trades = np.where(is_call_res, min_trades_call, min_trades_put)
        req_lev = np.where(is_call_res, min_lev_call, min_lev_put)

        eligible_lev_mask = (
            (~df_res["عمیقاً بی‌ارزش"])
            & (df_res["ارزش معاملات امروز (ریال)"] >= req_val)
            & (df_res["تعداد معاملات امروز"] >= req_trades)
            & (df_res["تعداد معاملات امروز"] > 0)
            & (df_res["روزهای تا سررسید (DTE)"] >= min_dte)
            & (df_res["اهرم"].notna())
            & (df_res["اهرم"] >= req_lev)
        )
        n_lev = int(eligible_lev_mask.sum())
        df_res["رتبه صدکی اهرم"] = 0.0
        if n_lev > 1:
            lev_ranks = rankdata(df_res.loc[eligible_lev_mask, "اهرم"], method="average")
            df_res.loc[eligible_lev_mask, "رتبه صدکی اهرم"] = np.round((lev_ranks / n_lev) * 100.0, 1)
        elif n_lev == 1:
            df_res.loc[eligible_lev_mask, "رتبه صدکی اهرم"] = 50.0

    else:
        df_res["رتبه صدکی حباب"] = 50.0
        df_res["جذابیت حباب (معکوس)"] = 50.0
        df_res["رتبه صدکی نقدینگی"] = 50.0
        df_res["رتبه صدکی اهرم"] = 50.0

    # =========================================================================
    # ۸. محاسبه امتیاز الگوریتم واحد (calculate_score) نسخه v3 برای تک‌تک قراردادها
    # =========================================================================
    from analytics.unified_scoring import calculate_score

    # محاسبه رتبه صدکی نقدینگی ساختاری ۵ روزه
    avg_5d_ranks = rankdata(df_res["میانگین ارزش ۵ روزه (ریال)"], method="average")
    trade_ranks = rankdata(df_res["تعداد معاملات امروز"], method="average")
    structural_pcts = ((avg_5d_ranks / n_items) * 0.70 + (trade_ranks / n_items) * 0.30) * 100.0

    scores = []
    readiness_list = []
    liq_list = []
    struct_liq_list = []
    spike_liq_list = []
    rel_val_list = []
    lev_score_list = []
    dte_list = []
    pass_hard_list = []
    explanations = []
    gate_mult_list = []

    for idx, (_, r) in enumerate(df_res.iterrows()):
        res = calculate_score(
            row=r,
            structural_liq_pct=structural_pcts[idx],
            bubble_pct_rank=df_res["رتبه صدکی حباب"].iloc[idx],
            config=config,
            leverage_pct_rank=df_res["رتبه صدکی اهرم"].iloc[idx],
        )
        scores.append(res["final_score"])
        readiness_list.append(res["readiness_score"])
        liq_list.append(res["combined_liquidity_score"])
        struct_liq_list.append(res["structural_liquidity_score"])
        spike_liq_list.append(res["spike_liquidity_score"])
        rel_val_list.append(res["relative_value_score"])
        lev_score_list.append(res["leverage_score"])
        dte_list.append(res["dte_suitability_score"])
        pass_hard_list.append(res["passes_hard_filter"])
        explanations.append(res["explanation"])
        gate_mult_list.append(res["liquidity_gate_multiplier"])

    df_res["امتیاز الگوریتم"] = scores
    df_res["امتیاز ترکیبی"] = scores
    df_res["امتیاز آمادگی پایه"] = readiness_list
    df_res["امتیاز نقدینگی ترکیبی"] = liq_list
    df_res["امتیاز نقدینگی ساختاری"] = struct_liq_list
    df_res["امتیاز جهش لحظه‌ای"] = spike_liq_list
    df_res["امتیاز ارزش نسبی (حباب)"] = rel_val_list
    df_res["امتیاز اهرم"] = lev_score_list
    df_res["امتیاز تناسب DTE"] = dte_list
    df_res["واجد فیلتر سخت"] = pass_hard_list
    df_res["چرا این امتیاز"] = explanations
    df_res["شرح فرمول امتیاز"] = explanations
    df_res["ضریب دروازه نقدینگی"] = gate_mult_list

    # مرتب‌سازی اصلی بر اساس بیشترین امتیاز الگوریتم (قراردادهای بدون امتیاز/Deep OTM در انتها)
    df_res = df_res.sort_values("امتیاز الگوریتم", ascending=False, na_position="last").reset_index(drop=True)
    logger.info(f"پردازش متریک‌ها به اتمام رسید. مجموع {len(df_res)} نماد با الگوریتم واحد نسخه v3 رتبه‌بندی شدند.")

    return df_res
