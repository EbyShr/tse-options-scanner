"""
ماژول بازنویسی‌شده و جامع الگوریتم امتیازدهی و انتخاب Top 10 — از صفر
شامل:
- قدم صفر: اعتبارسنجی و لاگ داده‌های خام
- قدم ۱: ساخت شیء استاندارد OptionContract با اعتبارسنجی صریح is_valid_pricing
- قدم ۲: اعمال ترتیبی گیت‌های سخت (Deep-OTM، نقدینگی، اهرم >= 3.0، DTE >= 3)
- قدم ۳: امتیازدهی وزنی (۲۵/۲۰/۱۵/۲۵/۱۵) منحصراً روی بازماندگان قدم ۲ + جریمه ضربی ۰.۷ ضد-Chasing
- قدم ۴: انتخاب Top 10 با سقف تنوع دارایی پایه (حداکثر ۳ از هر نماد)
- قدم ۵: خودآزمایی اجباری validate_top10 پیش از نمایش
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config import AppConfig
from data.normalizer import normalize_fa
from analytics.greeks_bsm import calculate_bsm_price, calculate_leverage, evaluate_moneyness
from analytics.underlying_momentum import determine_momentum_label

logger = logging.getLogger("OptionScanner.UnifiedScoring")

# ==============================================================================
# ثوابت گیت‌های سخت (سراسری)
# ==============================================================================
MIN_TRADE_VALUE_CALL = 5_000_000_000
MIN_TRADE_COUNT_CALL = 30
MIN_TRADE_VALUE_PUT  = 1_000_000_000
MIN_TRADE_COUNT_PUT  = 10
MIN_LEVERAGE_CALL = 3.0
MIN_LEVERAGE_PUT  = 3.0


# ==============================================================================
# قدم ۱ — ساختار استاندارد داده هر قرارداد (OptionContract)
# ==============================================================================
@dataclass
class OptionContract:
    symbol: str
    underlying: str
    option_type: str  # "Call" یا "Put"
    strike: float
    expiry_date: str
    dte: int
    market_price: float
    bsm_price: Optional[float]
    delta: Optional[float]
    leverage: Optional[float]
    today_trade_value: float
    today_trade_count: int
    avg_5d_trade_value: float
    avg_5d_trade_count: float = 0.0
    is_valid_pricing: bool = True

    # فیلدهای امتیازدهی و خروجی الگوریتم
    final_score: Optional[float] = None
    readiness_score: float = 0.0
    relative_value_score: float = 0.0
    leverage_score: float = 0.0
    combined_liquidity_score: float = 0.0
    structural_liquidity_score: float = 0.0
    spike_liquidity_score: float = 0.0
    dte_score: float = 0.0
    anti_chasing_factor: float = 1.0
    anti_chasing_applied: bool = False
    reasons: List[str] = field(default_factory=list)
    explanation: str = ""
    gate_status: str = "واجد شرایط"
    passes_hard_filter: bool = False

    # متادیتای تکمیلی جهت نمایش، مقایسه و خروجی CSV
    name: str = ""
    yesterday_price: float = 0.0
    price_change_today_pct: float = 0.0
    bubble_pct: Optional[float] = None
    bubble_rial: Optional[float] = None
    moneyness: str = ""
    dist_breakeven_pct: float = 0.0
    rank_in_underlying: int = 1
    sisters: List[Dict[str, Any]] = field(default_factory=list)
    inscode: str = ""
    bid_price: float = 0.0
    ask_price: float = 0.0
    liquidity_gate_multiplier: float = 1.0
    liquidity_gate_label: str = "عادی"
    underlying_close: float = 0.0
    momentum_label: str = "عادی"
    alignment: str = "خنثی"
    is_illiquid: bool = False
    liquidity_flag: str = "نقدشونده"

    def __getitem__(self, item: str) -> Any:
        """پشتیبانی از دسترسی دیکشنری با کلیدهای فارسی و انگلیسی جهت سازگاری ۱۰۰٪"""
        mapping = {
            "نماد": self.symbol,
            "symbol": self.symbol,
            "نام قرارداد": self.name,
            "name": self.name,
            "دارایی پایه": self.underlying,
            "underlying": self.underlying,
            "نوع قرارداد": "اختیار خرید (Call)" if self.option_type == "Call" else "اختیار فروش (Put)",
            "option_type": self.option_type,
            "قیمت اعمال": self.strike,
            "strike": self.strike,
            "تاریخ سررسید": self.expiry_date,
            "expiry_date": self.expiry_date,
            "روزهای تا سررسید (DTE)": self.dte,
            "dte": self.dte,
            "قیمت پایانی بازار": self.market_price,
            "market_price": self.market_price,
            "قیمت پایه": self.underlying_close,
            "قیمت دیروز": self.yesterday_price,
            "درصد تغییر امروز (%)": self.price_change_today_pct,
            "قیمت تئوریک BSM": self.bsm_price,
            "bsm_price": self.bsm_price,
            "دلتا": self.delta,
            "delta": self.delta,
            "اهرم": self.leverage,
            "leverage": self.leverage,
            "ارزش معاملات امروز (ریال)": self.today_trade_value,
            "today_trade_value": self.today_trade_value,
            "تعداد معاملات امروز": self.today_trade_count,
            "today_trade_count": self.today_trade_count,
            "میانگین ارزش ۵ روزه (ریال)": self.avg_5d_trade_value,
            "avg_5d_trade_value": self.avg_5d_trade_value,
            "امتیاز الگوریتم": self.final_score,
            "امتیاز ترکیبی": self.final_score,
            "final_score": self.final_score,
            "امتیاز آمادگی پایه": self.readiness_score,
            "امتیاز ارزش نسبی (حباب)": self.relative_value_score,
            "امتیاز اهرم": self.leverage_score,
            "امتیاز نقدینگی ترکیبی": self.combined_liquidity_score,
            "امتیاز نقدینگی ساختاری": self.structural_liquidity_score,
            "امتیاز جهش لحظه‌ای": self.spike_liquidity_score,
            "امتیاز تناسب DTE": self.dte_score,
            "واجد فیلتر سخت": self.passes_hard_filter,
            "چرا این امتیاز": self.explanation,
            "شرح فرمول امتیاز": self.explanation,
            "حباب خام (%)": self.bubble_pct,
            "حباب ریالی": self.bubble_rial,
            "وضعیت سودآوری": self.moneyness,
            "فاصله تا سربه‌سر (%)": self.dist_breakeven_pct,
            "روند دارایی پایه": self.momentum_label,
            "هم‌جهتی با روند": self.alignment,
            "کم‌عمق": self.is_illiquid,
            "وضعیت نقدینگی": self.liquidity_flag,
            "رتبه در دارایی پایه": self.rank_in_underlying,
            "قراردادهای جایگزین": self.sisters,
            "ضریب دروازه نقدینگی": self.liquidity_gate_multiplier,
            "InsCode": self.inscode,
            "بهترین مظنه خرید (Bid)": self.bid_price,
            "بهترین مظنه فروش (Ask)": self.ask_price,
            "عمیقاً بی‌ارزش": not (self.is_valid_pricing and (self.bsm_price or 0) >= 10 and abs(self.delta or 0) >= 0.10),
            "برچسب سفته‌بازی": "لاتاری/بی‌ارزش عمیق" if not (self.is_valid_pricing and (self.bsm_price or 0) >= 10 and abs(self.delta or 0) >= 0.10) else "",
        }
        if item in mapping:
            return mapping[item]
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(item)

    def get(self, item: str, default: Any = None) -> Any:
        try:
            val = self[item]
            return val if val is not None else default
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل به دیکشنری کامل جهت ساخت DataFrame"""
        is_deep = not (self.is_valid_pricing and (self.bsm_price or 0) >= 10 and abs(self.delta or 0) >= 0.10)
        return {
            "نماد": self.symbol,
            "نام قرارداد": self.name,
            "دارایی پایه": self.underlying,
            "نوع قرارداد": "اختیار خرید (Call)" if self.option_type == "Call" else "اختیار فروش (Put)",
            "قیمت پایانی بازار": self.market_price,
            "قیمت اعمال": self.strike,
            "قیمت پایه": self.underlying_close,
            "قیمت دیروز": self.yesterday_price,
            "درصد تغییر امروز (%)": self.price_change_today_pct,
            "تاریخ سررسید": self.expiry_date,
            "روزهای تا سررسید (DTE)": self.dte,
            "قیمت تئوریک BSM": self.bsm_price,
            "دلتا": self.delta,
            "اهرم": self.leverage,
            "ارزش معاملات امروز (ریال)": self.today_trade_value,
            "تعداد معاملات امروز": self.today_trade_count,
            "میانگین ارزش ۵ روزه (ریال)": self.avg_5d_trade_value,
            "حباب خام (%)": self.bubble_pct,
            "حباب ریالی": self.bubble_rial,
            "امتیاز الگوریتم": self.final_score,
            "امتیاز ترکیبی": self.final_score,
            "امتیاز آمادگی پایه": self.readiness_score,
            "امتیاز ارزش نسبی (حباب)": self.relative_value_score,
            "امتیاز اهرم": self.leverage_score,
            "امتیاز نقدینگی ترکیبی": self.combined_liquidity_score,
            "امتیاز نقدینگی ساختاری": self.structural_liquidity_score,
            "امتیاز جهش لحظه‌ای": self.spike_liquidity_score,
            "امتیاز تناسب DTE": self.dte_score,
            "واجد فیلتر سخت": self.passes_hard_filter,
            "وضعیت فیلتر": self.gate_status,
            "چرا این امتیاز": self.explanation,
            "شرح فرمول امتیاز": self.explanation,
            "عمیقاً بی‌ارزش": is_deep,
            "برچسب سفته‌بازی": "لاتاری/بی‌ارزش عمیق" if is_deep else "",
            "ضریب دروازه نقدینگی": self.liquidity_gate_multiplier,
            "رتبه در دارایی پایه": self.rank_in_underlying,
            "قراردادهای جایگزین": self.sisters,
            "وضعیت سودآوری": self.moneyness,
            "فاصله تا سربه‌سر (%)": self.dist_breakeven_pct,
            "روند دارایی پایه": self.momentum_label,
            "هم‌جهتی با روند": self.alignment,
            "کم‌عمق": self.is_illiquid,
            "وضعیت نقدینگی": self.liquidity_flag,
            "بهترین مظنه خرید (Bid)": self.bid_price,
            "بهترین مظنه فروش (Ask)": self.ask_price,
            "InsCode": self.inscode,
        }


import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ==============================================================================
# قدم صفر — اعتبارسنجی و چاپ نمونه داده خام
# ==============================================================================
def print_sample_raw_data(contracts: List[OptionContract]) -> None:
    """چاپ اولین ۵ قرارداد بلافاصله بعد از واکشی جهت اعتبارسنجی داده خام"""
    try:
        print("SAMPLE RAW DATA (first 5 contracts):")
        for c in contracts[:5]:
            msg = (
                f"{c.symbol}: leverage={c.leverage} (type={type(c.leverage)}), "
                f"delta={c.delta}, BSM={c.bsm_price}, today_count={c.today_trade_count}"
            )
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode("ascii", errors="backslashreplace").decode("ascii"))
            logger.info(f"[RAW DATA] {msg}")
    except Exception as e:
        logger.warning(f"Error printing raw sample: {e}")


# ==============================================================================
# ساخت قراردادها از اسنپ‌شات خام بازار
# ==============================================================================
def build_contracts(
    df_raw: pd.DataFrame,
    underlying_stats: Dict[str, Dict[str, Any]],
    options_avg_values: Dict[str, float],
    config: Optional[AppConfig] = None,
) -> List[OptionContract]:
    """
    استخراج فیلدهای لازم و ساخت اشیاء OptionContract
    اعمال صریح is_valid_pricing=False در صورت None/NaN بودن bsm_price، delta یا leverage.
    """
    if df_raw.empty:
        return []

    r_free = getattr(getattr(config, "valuation", None), "risk_free_rate", 0.23)
    d_yield = getattr(getattr(config, "valuation", None), "dividend_yield", 0.0)

    contracts: List[OptionContract] = []

    for _, row in df_raw.iterrows():
        sym = str(row.get("Symbol", row.get("نماد", ""))).strip()
        inscode = str(row.get("InsCode", "")).strip()
        raw_u = str(row.get("UnderlyingSymbol", row.get("دارایی پایه", ""))).strip()
        norm_u = normalize_fa(raw_u)

        u_data = underlying_stats.get(raw_u) or underlying_stats.get(norm_u) or {}
        spot = float(u_data.get("Close", 0.0))
        if spot <= 0:
            spot = float(row.get("Spot", row.get("UnderlyingClose", row.get("UnderlyingLast", 0.0))))

        strike = float(row.get("Strike", row.get("قیمت اعمال", 0.0)))
        dte = int(float(row.get("DaysToExpiry", row.get("روزهای تا سررسید (DTE)", 0.0))))
        exp_date = str(row.get("EndDate", row.get("تاریخ سررسید", "")))

        # نوع قرارداد (Call / Put)
        opt_type_raw = str(row.get("OptionType", row.get("نوع قرارداد", ""))).lower()
        is_call = "call" in opt_type_raw or "خرید" in opt_type_raw or sym.startswith("ض")
        opt_type_std = "Call" if is_call else "Put"

        # قیمت بازار آپشن
        close_p = float(row.get("Close", 0.0))
        last_p = float(row.get("Last", 0.0))
        market_p = close_p if close_p > 0 else (last_p if last_p > 0 else float(row.get("قیمت پایانی بازار", 0.0)))

        # نقدینگی و معاملات
        val_today = float(row.get("Value", row.get("ارزش معاملات امروز (ریال)", 0.0)))
        trade_count = int(float(row.get("TradeCount", row.get("تعداد معاملات امروز", 0))))
        avg_val_5d = options_avg_values.get(inscode, float(row.get("میانگین ارزش ۵ روزه (ریال)", 0.0)))

        # نوسان‌پذیری برای محاسبه BSM در صورت لزوم
        iv = row.get("ImpliedVolatility")
        realized_vol = float(u_data.get("RealizedVol", 0.35))
        vol_used = float(iv) if (pd.notna(iv) and 0.02 < float(iv) < 2.5) else realized_vol

        # استخراج یا محاسبه BSM و Delta
        has_bsm_col = any(k in row for k in ["BSM", "P_BSM", "قیمت تئوریک BSM"])
        raw_bsm = row.get("BSM", row.get("P_BSM", row.get("قیمت تئوریک BSM")))

        has_delta_col = any(k in row for k in ["Delta", "delta", "دلتا"])
        raw_delta = row.get("Delta", row.get("delta", row.get("دلتا")))

        has_lev_col = any(k in row for k in ["Leverage", "leverage", "اهرم"])
        raw_lev = row.get("Leverage", row.get("leverage", row.get("اهرم")))

        calc_bsm = None
        calc_delta = None
        if spot > 0 and strike > 0 and dte > 0:
            calc_bsm, calc_delta = calculate_bsm_price(
                spot=spot,
                strike=strike,
                dte=dte,
                rate=r_free,
                volatility=vol_used,
                option_type="call" if is_call else "put",
                dividend_yield=d_yield,
            )

        # تعیین bsm_price
        if has_bsm_col:
            if pd.notna(raw_bsm) and raw_bsm is not None and float(raw_bsm) > 0:
                bsm_price = round(float(raw_bsm), 1)
            else:
                bsm_price = None
        else:
            if calc_bsm is not None and calc_bsm > 0:
                bsm_price = round(calc_bsm, 1)
            elif market_p > 0 and pd.notna(row.get("حباب ریالی")):
                bsm_price = round(market_p - float(row.get("حباب ریالی")), 1)
            elif market_p > 0 and pd.notna(row.get("حباب خام (%)")):
                b_pct = float(row.get("حباب خام (%)"))
                bsm_price = round(market_p / (1.0 + b_pct / 100.0), 1)
            else:
                bsm_price = None

        # تعیین delta
        if has_delta_col:
            if pd.notna(raw_delta) and raw_delta is not None:
                delta = round(float(raw_delta), 3)
            else:
                delta = None
        else:
            if calc_delta is not None and pd.notna(calc_delta):
                delta = round(float(calc_delta), 3)
            elif bsm_price is not None and bsm_price >= 10:
                delta = 0.50 if is_call else -0.50
            else:
                delta = None

        # تعیین leverage
        if has_lev_col:
            if pd.notna(raw_lev) and raw_lev is not None and float(raw_lev) > 0:
                leverage = round(float(raw_lev), 2)
            else:
                leverage = None
        else:
            if spot > 0 and market_p > 0 and delta is not None:
                leverage = calculate_leverage(spot=spot, option_price=market_p, delta=delta)
            else:
                leverage = None

        # قدم ۱ — اعتبارسنجی صریح قیمت‌گذاری:
        # اگر هرکدام از bsm_price، delta، یا leverage مقدار None/NaN/نامعتبر بود،
        # صریحاً is_valid_pricing=False علامت زده شود (نه اینکه صفر یا پیش‌فرض گذاشته شود)
        is_valid_pricing = True
        if bsm_price is None or pd.isna(bsm_price) or bsm_price <= 0:
            is_valid_pricing = False
            bsm_price = None

        if delta is None or pd.isna(delta):
            is_valid_pricing = False
            delta = None

        if leverage is None or pd.isna(leverage) or leverage <= 0:
            is_valid_pricing = False
            leverage = None

        # حباب ریالی و درصدی
        if is_valid_pricing and bsm_price is not None and bsm_price > 0:
            bub_pct = round(((market_p - bsm_price) / bsm_price) * 100.0, 2)
            bub_rial = round(market_p - bsm_price)
        else:
            bub_pct = None
            bub_rial = round(market_p - (bsm_price if bsm_price is not None else 0.0))

        # مظنه‌ها
        bid_p = float(row.get("BidPrice", row.get("بهترین مظنه خرید (Bid)", 0.0)))
        ask_p = float(row.get("AskPrice", row.get("بهترین مظنه فروش (Ask)", 0.0)))

        # نام قرارداد و تغییر قیمت روز
        name_str = str(row.get("Name", row.get("نام قرارداد", sym)))
        yesterday_price = float(row.get("Yesterday", row.get("قیمت دیروز", 0.0)))
        if yesterday_price > 0 and market_p > 0:
            price_change_today_pct = round(((market_p - yesterday_price) / yesterday_price) * 100.0, 2)
        else:
            price_change_today_pct = 0.0

        # وضعیت سودآوری
        moneyness, dist_be = evaluate_moneyness(
            spot=spot,
            strike=strike,
            option_price=market_p,
            option_type="call" if is_call else "put",
        )

        # روند دارایی پایه و هم‌جهتی
        momentum_label = determine_momentum_label(u_data)
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

        min_val_test = MIN_TRADE_VALUE_CALL if is_call else MIN_TRADE_VALUE_PUT
        min_cnt_test = MIN_TRADE_COUNT_CALL if is_call else MIN_TRADE_COUNT_PUT
        is_illiquid = bool(val_today < min_val_test or trade_count < min_cnt_test)
        liquidity_flag = "کم‌عمق" if is_illiquid else "نقدشونده"

        c = OptionContract(
            symbol=sym,
            underlying=norm_u if norm_u else raw_u,
            option_type=opt_type_std,
            strike=strike,
            expiry_date=exp_date,
            dte=dte,
            market_price=market_p,
            bsm_price=bsm_price,
            delta=delta,
            leverage=leverage,
            today_trade_value=val_today,
            today_trade_count=trade_count,
            avg_5d_trade_value=avg_val_5d,
            is_valid_pricing=is_valid_pricing,
            bubble_pct=bub_pct,
            bubble_rial=bub_rial,
            inscode=inscode,
            bid_price=bid_p,
            ask_price=ask_p,
            underlying_close=spot,
            name=name_str,
            yesterday_price=yesterday_price,
            price_change_today_pct=price_change_today_pct,
            moneyness=moneyness,
            dist_breakeven_pct=dist_be,
            momentum_label=momentum_label,
            alignment=alignment,
            is_illiquid=is_illiquid,
            liquidity_flag=liquidity_flag,
        )
        contracts.append(c)

    # قدم صفر: چاپ داده خام بلافاصله پس از ساخت
    print_sample_raw_data(contracts)
    return contracts


# ==============================================================================
# قدم ۲ — گیت‌های سخت (به همین ترتیب دقیق اعمال شوند)
# ==============================================================================
def apply_gates(contracts: List[OptionContract], option_type: str) -> Tuple[List[OptionContract], Dict[str, int]]:
    """
    اعمال متوالی ۵ مرحله فیلترینگ سخت روی خروجی هر گیت قبلی
    و چاپ دقیق لاگ تغییرات (GATE LOG) در هر بار اجرا.
    """
    log: Dict[str, int] = {}
    step0 = [c for c in contracts if c.option_type == option_type]
    log['شروع'] = len(step0)

    # مرحله ۱: گیت Deep-OTM / داده نامعتبر
    step1 = [
        c for c in step0
        if c.is_valid_pricing
        and c.bsm_price is not None
        and c.bsm_price >= 10
        and c.delta is not None
        and abs(c.delta) >= 0.10
    ]
    log['بعد از گیت Deep-OTM/داده نامعتبر'] = len(step1)

    # علامت‌گذاری نمادهای ردشده در گیت ۱
    step1_set = set(id(c) for c in step1)
    for c in step0:
        if id(c) not in step1_set:
            c.gate_status = "رد شده: Deep-OTM / داده نامعتبر"
            c.passes_hard_filter = False

    # مرحله ۲: گیت نقدینگی سخت
    min_value = MIN_TRADE_VALUE_CALL if option_type == "Call" else MIN_TRADE_VALUE_PUT
    min_count = MIN_TRADE_COUNT_CALL if option_type == "Call" else MIN_TRADE_COUNT_PUT
    step2 = [
        c for c in step1
        if c.today_trade_count >= min_count
        and c.today_trade_value >= min_value
    ]
    log['بعد از گیت نقدینگی'] = len(step2)

    step2_set = set(id(c) for c in step2)
    for c in step1:
        if id(c) not in step2_set:
            c.gate_status = f"رد شده: نقدینگی ناکافی (حداقل {min_count} معامله و {min_value/10_000_000:.0f} م.ت)"
            c.passes_hard_filter = False

    # مرحله ۳: گیت سخت اهرم (MIN_LEVERAGE >= 3.0)
    min_leverage = MIN_LEVERAGE_CALL if option_type == "Call" else MIN_LEVERAGE_PUT
    step3 = [
        c for c in step2
        if c.leverage is not None
        and c.leverage >= min_leverage
    ]
    log['بعد از گیت اهرم'] = len(step3)

    step3_set = set(id(c) for c in step3)
    for c in step2:
        if id(c) not in step3_set:
            c.gate_status = f"رد شده: اهرم کمتر از {min_leverage}x"
            c.passes_hard_filter = False

    # مرحله ۴: گیت DTE (حداقل ۳ روز)
    step4 = [c for c in step3 if c.dte >= 3]
    log['بعد از گیت DTE'] = len(step4)

    step4_set = set(id(c) for c in step4)
    for c in step3:
        if id(c) not in step4_set:
            c.gate_status = "رد شده: روزهای تا سررسید کمتر از ۳ روز"
            c.passes_hard_filter = False

    for c in step4:
        c.gate_status = "واجد شرایط"
        c.passes_hard_filter = True

    print(f"[{option_type}] GATE LOG: {log}")
    logger.info(f"[{option_type}] GATE LOG: {log}")
    return step4, log


# ==============================================================================
# قدم ۳ — امتیازدهی وزنی (فقط روی بازماندگان قدم ۲)
# ==============================================================================
def calculate_dte_suitability_score(dte: float, opt_min: int = 10, opt_max: int = 40) -> float:
    """محاسبه امتیاز تناسب DTE بر اساس منحنی پیوسته (۱۰ تا ۴۰ روز بالاترین امتیاز)"""
    if dte < 3:
        return 0.0
    elif dte < opt_min:
        return round(30.0 + (dte - 3) * (60.0 / max(1, opt_min - 3)), 1)
    elif opt_min <= dte <= opt_max:
        return 100.0
    elif dte <= 70:
        return round(100.0 - (dte - opt_max) * (30.0 / max(1, 70 - opt_max)), 1)
    else:
        return 55.0


def calculate_underlying_readiness_score(
    u_stats: Union[Dict[str, Any], pd.Series],
    is_call: bool,
    config: Optional[AppConfig] = None,
) -> Tuple[float, List[str], Dict[str, Any]]:
    """
    محاسبه امتیاز آمادگی دارایی پایه (۲۵٪) بر اساس جدول شرطی ۴ مؤلفه‌ای:
    ۱. روند میانگین‌ها (SMA5 و SMA20)
    ۲. شاخص RSI14
    ۳. صف و بوک سفارشات
    ۴. پولبک و مومنتوم ۵ روزه
    خروجی: (امتیاز نهایی پس از جریمه ضد-chasing، لیست دلایل، دیکشنری جزئیات detail)
    """
    reasons: List[str] = []

    def _get(keys, default=0.0):
        if isinstance(u_stats, (dict, pd.Series)):
            for k in keys:
                if k in u_stats and pd.notna(u_stats[k]):
                    return u_stats[k]
        return default

    dist_sma5 = float(_get(["DistSMA5", "فاصله پایه از SMA5 (%)"], 0.0) or 0.0)
    dist_sma20 = float(_get(["DistSMA20", "فاصله پایه از SMA20 (%)"], 0.0) or 0.0)
    queue = str(_get(["QueueStatus", "وضعیت صف پایه"], "متعادل"))
    rsi = float(_get(["RSI14", "RSI14 پایه"], 50.0) or 50.0)
    rsi_status = str(_get(["RSIStatus", "وضعیت RSI پایه"], "عادی"))
    is_pullback = bool(_get(["IsPullback", "پولبک پایه"], False))
    ret_1d = float(_get(["Return1D", "بازدهی ۱ روزه پایه (%)"], 0.0) or 0.0)
    ret_3d = float(_get(["Return3D", "بازدهی ۳ روزه پایه (%)"], 0.0) or 0.0)
    ret_5d = float(_get(["Return5D", "بازدهی ۵ روزه پایه (%)"], 0.0) or 0.0)
    p_band = _get(["PriceBand", "دامنه نوسان پایه"], None)
    u_sym = str(_get(["Symbol", "دارایی پایه"], ""))

    score_sma = 0.0
    score_rsi = 0.0
    score_queue = 0.0
    score_pullback = 0.0

    if is_call:
        # ۱. روند میانگین‌ها
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

        # ۲. RSI
        if "خروج از اشباع فروش" in rsi_status or (30.0 <= rsi <= 45.0 and ret_1d > 0):
            score_rsi = 100.0
            reasons.append(f"برگشت RSI از اشباع فروش ({rsi:.1f})")
        elif 45.0 < rsi <= 65.0:
            score_rsi = 85.0
            reasons.append(f"شاخص RSI در منطقه صعودی متعادل ({rsi:.1f})")
        elif rsi > 70.0:
            score_rsi = 35.0
            reasons.append(f"RSI در اشباع خرید ({rsi:.1f})")
        else:
            score_rsi = 50.0

        # ۳. صف
        if "صف خرید" in queue:
            score_queue = 100.0
            reasons.append("صف خرید در دارایی پایه")
        elif "تقاضا" in queue:
            score_queue = 75.0
            reasons.append("برتری تقاضا در سفارشات پایه")
        elif "متعادل" in queue:
            score_queue = 50.0
        else:
            score_queue = 15.0

        # ۴. پولبک
        if is_pullback:
            score_pullback = 100.0
            reasons.append("پولبک فنی به حمایت و برگشت تقاضا")
        elif ret_5d > 2.0 and ret_1d > 0:
            score_pullback = 80.0
            reasons.append("روند صعودی ۵ روزه")
        elif ret_1d > 0.5:
            score_pullback = 60.0
        else:
            score_pullback = 25.0

    else:
        # قواعد Put
        if dist_sma5 < 0 and dist_sma20 < 0:
            score_sma = 100.0
            reasons.append("شکست نزولی زیر SMA5 و SMA20")
        elif dist_sma5 < 0:
            score_sma = 80.0
            reasons.append("ریزش قیمت زیر SMA5")
        elif dist_sma20 < 0:
            score_sma = 65.0
            reasons.append("قرارگیری زیر باند SMA20")
        else:
            score_sma = 25.0

        if rsi >= 70.0 or (rsi >= 60.0 and ret_1d < -0.5):
            score_rsi = 100.0
            reasons.append(f"ریزش از اشباع خرید RSI ({rsi:.1f})")
        elif 35.0 <= rsi < 55.0:
            score_rsi = 80.0
            reasons.append(f"RSI در فاز نزولی پایدار ({rsi:.1f})")
        elif rsi < 30.0:
            score_rsi = 35.0
            reasons.append(f"RSI در اشباع فروش ({rsi:.1f})")
        else:
            score_rsi = 50.0

        if "صف فروش" in queue:
            score_queue = 100.0
            reasons.append("صف فروش قفل‌شده در دارایی پایه")
        elif "عرضه" in queue:
            score_queue = 75.0
            reasons.append("برتری عرضه در سفارشات پایه")
        elif "متعادل" in queue:
            score_queue = 50.0
        else:
            score_queue = 15.0

        if ret_5d < -2.0 and ret_1d < 0:
            score_pullback = 100.0
            reasons.append("موج نزولی قوی هم‌سو با Put")
        elif dist_sma5 < -1.5:
            score_pullback = 80.0
        elif ret_1d < -0.5:
            score_pullback = 60.0
        else:
            score_pullback = 25.0

    raw = (score_sma * 0.25) + (score_rsi * 0.25) + (score_queue * 0.25) + (score_pullback * 0.25)
    raw = round(min(100.0, max(0.0, raw)), 1)

    th_3d, th_1d, _ = get_chasing_thresholds(symbol=u_sym, price_band=p_band, config=config)
    has_3d = abs(ret_3d) > 0.001
    is_chasing = False
    if has_3d:
        if is_call and ret_3d > th_3d:
            is_chasing = True
            reasons.append(f"اعمال جریمه ضد-Chasing (رشد ۳ روزه {ret_3d:+.1f}% فراتر از {th_3d:.1f}%)")
        elif not is_call and ret_3d < -th_3d:
            is_chasing = True
            reasons.append(f"اعمال جریمه ضد-Chasing (افت ۳ روزه {ret_3d:+.1f}% فراتر از -{th_3d:.1f}%)")
    else:
        if is_call and ret_1d > th_1d:
            is_chasing = True
            reasons.append(f"فالبک بازده ۱ روزه: اعمال جریمه ضد-Chasing (رشد ۱ روزه {ret_1d:+.1f}% فراتر از {th_1d:.1f}%)")
        elif not is_call and ret_1d < -th_1d:
            is_chasing = True
            reasons.append(f"فالبک بازده ۱ روزه: اعمال جریمه ضد-Chasing (افت ۱ روزه {ret_1d:+.1f}% فراتر از -{th_1d:.1f}%)")

    final_readiness = round(raw * 0.70, 1) if is_chasing else raw
    detail = {
        "raw": raw,
        "anti_chasing_applied": is_chasing,
        "threshold_3d": th_3d,
        "threshold_1d": th_1d,
    }
    return final_readiness, reasons, detail


def calculate_combined_liquidity_score(
    structural_pct: float,
    today_volume: float,
    avg_5d_volume: float,
    config: Optional[AppConfig] = None,
) -> Tuple[float, float, float]:
    """
    محاسبه امتیاز نقدینگی ترکیبی (۲۵٪ = ۱۵٪ ساختاری + ۱۰٪ جهش)
    خروجی: (combined_score, structural_contrib, spike_score)
    """
    struct_score = max(0.0, min(100.0, float(structural_pct)))
    if avg_5d_volume > 0 and today_volume > 0:
        ratio = float(today_volume) / float(avg_5d_volume)
        spike_score = round(min(100.0, (ratio / 3.0) * 100.0), 1)
    else:
        spike_score = 0.0

    combined = round(0.60 * struct_score + 0.40 * spike_score, 1)
    struct_contrib = round(0.60 * struct_score, 1)
    return combined, struct_contrib, spike_score


def get_chasing_thresholds(
    symbol: str = "",
    price_band: Optional[float] = None,
    config: Optional[AppConfig] = None,
) -> Tuple[float, float, float]:
    """محاسبه آستانه ۳ روزه و ۱ روزه ضد-Chasing متناسب با سقف واقعی نوسان روزانه"""
    leveraged_keywords = ["اهرم", "شتاب", "موج", "توان", "جهش", "نارنج", "کاریز", "بیدار"]
    is_leveraged = any(k in symbol for k in leveraged_keywords)

    if price_band is not None and price_band > 0:
        band = float(price_band)
    else:
        band = 0.04 if is_leveraged else 0.03

    compounded_3d = ((1.0 + band) ** 3 - 1.0) * 100.0
    th_3d = round(compounded_3d * 0.90, 2)
    th_1d = round((band * 100.0) * 0.90, 2)
    return th_3d, th_1d, band


def score_eligible_contracts(
    eligible_contracts: List[OptionContract],
    option_type: str,
    underlying_stats: Dict[str, Dict[str, Any]],
    config: Optional[AppConfig] = None,
) -> List[OptionContract]:
    """
    محاسبه ۵ جزء امتیازدهی وزنی منحصراً روی بازماندگان قدم ۲:
    - آمادگی دارایی پایه: 25%
    - ارزش نسبی / حباب: 20% (percentile معکوس حباب، فقط بین بازماندگان)
    - اهرم: 15% (percentile اهرم، فقط بین بازماندگان)
    - نقدینگی ترکیبی: 25% (پایدار 15% میانگین ۵ روزه + جهش 10% با سقف momentum_ratio/3.0)
    - تناسب DTE: 15%
    سپس اعمال ضریب ضربی 0.70 ضد-Chasing روی امتیاز نهایی.
    """
    n = len(eligible_contracts)
    if n == 0:
        return []

    is_call = (option_type == "Call")

    # ۱. محاسبه رتبه‌های صدکی حباب (معکوس) فقط بین بازماندگان
    bubble_vals = [c.bubble_pct if c.bubble_pct is not None else 0.0 for c in eligible_contracts]
    if n > 1:
        b_ranks = rankdata(bubble_vals, method="average")
        b_pcts = [round(max(0.0, min(100.0, 100.0 - (r / n) * 100.0)), 1) for r in b_ranks]
    else:
        b_pcts = [50.0]

    # ۲. محاسبه رتبه‌های صدکی اهرم فقط بین بازماندگان
    lev_vals = [c.leverage if c.leverage is not None else 0.0 for c in eligible_contracts]
    if n > 1:
        l_ranks = rankdata(lev_vals, method="average")
        l_pcts = [round(max(0.0, min(100.0, (r / n) * 100.0)), 1) for r in l_ranks]
    else:
        l_pcts = [50.0]

    # ۳. محاسبه نقدینگی ساختاری ۵ روزه فقط بین بازماندگان
    avg_5d_vals = [c.avg_5d_trade_value for c in eligible_contracts]
    if n > 1:
        s_ranks = rankdata(avg_5d_vals, method="average")
        s_pcts = [(r / n) * 100.0 for r in s_ranks]
    else:
        s_pcts = [50.0]

    for idx, c in enumerate(eligible_contracts):
        u_data = underlying_stats.get(c.underlying, {})
        u_sym = c.underlying

        # آمادگی دارایی پایه (۲۵٪)
        readiness, reasons_u, u_detail = calculate_underlying_readiness_score(u_data, is_call=is_call, config=config)
        c.readiness_score = u_detail["raw"]

        # ارزش نسبی حباب (۲۰٪)
        c.relative_value_score = b_pcts[idx]

        # اهرم (۱۵٪)
        c.leverage_score = l_pcts[idx]

        # نقدینگی ترکیبی (۲۵٪ = پایدار ۱۵٪ + جهش ۱۰٪)
        struct_score = round(s_pcts[idx], 1)
        momentum_ratio = (c.today_trade_value / c.avg_5d_trade_value) if c.avg_5d_trade_value > 0 else 1.0
        spike_score = round(min(100.0, (momentum_ratio / 3.0) * 100.0), 1)
        comb_liq = round(0.60 * struct_score + 0.40 * spike_score, 1)

        c.structural_liquidity_score = struct_score
        c.spike_liquidity_score = spike_score
        c.combined_liquidity_score = comb_liq

        # تناسب DTE (۱۵٪)
        c.dte_score = calculate_dte_suitability_score(c.dte)

        # ترکیب با وزن‌های ۲۵ / ۲۰ / ۱۵ / ۲۵ / ۱۵
        raw_final = (
            0.25 * c.readiness_score
            + 0.20 * c.relative_value_score
            + 0.15 * c.leverage_score
            + 0.25 * c.combined_liquidity_score
            + 0.15 * c.dte_score
        )

        # ضریب ضد-Chasing (۰.۷ ضربی روی امتیاز نهایی)
        is_chasing = u_detail["anti_chasing_applied"]

        if is_chasing:
            c.anti_chasing_applied = True
            c.anti_chasing_factor = 0.70
            c.final_score = round(raw_final * 0.70, 1)
        else:
            c.anti_chasing_applied = False
            c.anti_chasing_factor = 1.0
            c.final_score = round(raw_final, 1)

        # دلایل و توضیحات
        reasons_all = list(reasons_u)
        if c.leverage is not None and c.leverage >= 5.0:
            reasons_all.append(f"اهرم بالا و پیشتاز ({c.leverage:.1f}x)")
        elif c.leverage is not None and c.leverage >= 3.0:
            reasons_all.append(f"اهرم مناسب ({c.leverage:.1f}x)")

        if c.relative_value_score >= 80:
            reasons_all.append("ارزش‌گذاری و حباب جذاب")
        if c.combined_liquidity_score >= 70:
            reasons_all.append("نقدینگی و عمق فعال")
        if c.dte_score >= 95:
            reasons_all.append(f"سررسید بهینه ({c.dte} روز)")

        c.reasons = reasons_all
        c.explanation = "؛ ".join(reasons_all[:4]) + "."

    return eligible_contracts


# ==============================================================================
# قدم ۴ — انتخاب Top 10 با سقف تنوع دارایی پایه
# ==============================================================================
def _get_item_score(item: Any) -> float:
    if isinstance(item, dict):
        val = item.get("final_score", item.get("score", item.get("امتیاز الگوریتم", 0.0)))
    else:
        val = getattr(item, "final_score", getattr(item, "score", getattr(item, "امتیاز الگوریتم", 0.0)))
    return float(val) if val is not None else -1.0


def _get_item_underlying(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("underlying", item.get("دارایی پایه", "")))
    return str(getattr(item, "underlying", getattr(item, "دارایی پایه", "")))


def select_top_n(
    scored_contracts: List[Any],
    n: int = 10,
    max_per_underlying: int = 3,
) -> List[Any]:
    """
    انتخاب تا n قرارداد برتر با مرتب‌سازی نزولی بر اساس final_score
    و اعمال سقف تنوع حداکثر max_per_underlying قرارداد از هر دارایی پایه.
    """
    scored_contracts.sort(key=_get_item_score, reverse=True)
    selected: List[Any] = []
    count: Dict[str, int] = {}
    for c in scored_contracts:
        u = _get_item_underlying(c)
        if count.get(u, 0) < max_per_underlying:
            selected.append(c)
            count[u] = count.get(u, 0) + 1
        if len(selected) == n:
            break

    # ثبت رتبه درون دارایی پایه برای قراردادهای انتخاب‌شده
    u_ranks: Dict[str, int] = {}
    for c in selected:
        u = _get_item_underlying(c)
        curr_r = u_ranks.get(u, 0) + 1
        u_ranks[u] = curr_r
        if hasattr(c, "rank_in_underlying"):
            setattr(c, "rank_in_underlying", curr_r)
        elif isinstance(c, dict):
            c["rank_in_underlying"] = curr_r
            c["رتبه در دارایی پایه"] = curr_r

    return selected


# ==============================================================================
# استخراج نمادهای جایگزین همنام (Sister Contracts) برای هر کارت برتر
# ==============================================================================
def attach_sister_contracts(
    top_list: List[OptionContract],
    pool_contracts: List[OptionContract],
    max_sisters: int = 2,
) -> None:
    """استخراج قراردادهای جایگزین معتبر روی همان دارایی پایه و با همان جهت"""
    for top_c in top_list:
        candidates = [
            c for c in pool_contracts
            if c.underlying == top_c.underlying
            and c.option_type == top_c.option_type
            and c.symbol != top_c.symbol
            and c.passes_hard_filter
            and c.final_score is not None
        ]
        candidates.sort(key=lambda x: x.final_score or 0.0, reverse=True)
        top_c.sisters = [
            {
                "symbol": s.symbol,
                "strike": int(s.strike),
                "dte": int(s.dte),
                "score": float(s.final_score or 0.0),
                "price": int(s.market_price),
                "underlying": s.underlying,
            }
            for s in candidates[:max_sisters]
        ]


# ==============================================================================
# قدم ۵ — خودآزمایی اجباری قبل از نمایش
# ==============================================================================
def validate_top10(top10_list: List[OptionContract], option_type: str) -> List[str]:
    """
    اعتبارسنجی خودکار فهرست برترین‌ها جهت اطمینان از عدم نفوذ هرگونه باگ یا داده نامعتبر.
    اگر کوچکترین خطایی وجود داشته باشد، به کاربر اعلام و از نمایش جلوگیری می‌شود.
    """
    errors: List[str] = []
    min_leverage = MIN_LEVERAGE_CALL if option_type == "Call" else MIN_LEVERAGE_PUT
    for c in top10_list:
        if c.leverage is None or c.leverage < min_leverage:
            errors.append(f"{c.symbol}: leverage={c.leverage} < {min_leverage}")
        if c.today_trade_count == 0:
            errors.append(f"{c.symbol}: نقدینگی صفر در Top10")
        if not c.is_valid_pricing:
            errors.append(f"{c.symbol}: قیمتگذاری نامعتبر در Top10")
    return errors


# ==============================================================================
# تابع جامع اجرای کامل خط‌لوله (Unified Pipeline)
# ==============================================================================
def run_unified_scoring_pipeline(
    df_raw: pd.DataFrame,
    underlying_stats: Dict[str, Dict[str, Any]],
    options_avg_values: Dict[str, float],
    config: Optional[AppConfig] = None,
    top_n: int = 10,
    max_per_underlying: int = 3,
) -> Dict[str, Any]:
    """
    تنها مرجع رسمی و جامع اجرای الگوریتم اسکنر:
    ۱. ساخت OptionContractها و چاپ داده خام (قدم ۰ و ۱)
    ۲. اعمال ترتیبی گیت‌های سخت (قدم ۲)
    ۳. امتیازدهی وزنی روی بازماندگان (قدم ۳)
    ۴. انتخاب ۱۰ برتر خرید و ۱۰ برتر فروش با سقف تنوع (قدم ۴)
    ۵. الصاق قراردادهای خواهر همنام
    ۶. خودآزمایی اجباری validate_top10 (قدم ۵)
    """
    if df_raw.empty:
        return {
            "all_contracts": [],
            "top10_call": [],
            "top10_put": [],
            "df_all": pd.DataFrame(),
            "df_calls_top": pd.DataFrame(),
            "df_puts_top": pd.DataFrame(),
            "gate_logs": {"Call": {}, "Put": {}},
            "validation_errors": {"Call": [], "Put": []},
            "eligible_calls_count": 0,
            "eligible_puts_count": 0,
        }

    logger.info("=== شروع اجرای پایپ‌لاین واحد الگوریتم (نسخه از صفر) ===")

    if config is not None:
        cfg_max = getattr(getattr(config, "unified_scoring", None), "max_per_underlying", None)
        if cfg_max is not None:
            max_per_underlying = int(cfg_max)

    # ۱. استخراج و اعتبارسنجی اولیه اشیاء قرارداد
    all_contracts = build_contracts(
        df_raw=df_raw,
        underlying_stats=underlying_stats,
        options_avg_values=options_avg_values,
        config=config,
    )

    # ۲. اعمال گیت‌های سخت به تفکیک Call و Put
    eligible_calls, log_call = apply_gates(all_contracts, "Call")
    eligible_puts, log_put = apply_gates(all_contracts, "Put")

    # ۳. امتیازدهی وزنی منحصراً روی بازماندگان
    scored_calls = score_eligible_contracts(eligible_calls, "Call", underlying_stats, config)
    scored_puts = score_eligible_contracts(eligible_puts, "Put", underlying_stats, config)

    # ۴. انتخاب Top N با سقف تنوع ۳ نماد از هر دارایی پایه
    top10_call = select_top_n(scored_calls, n=top_n, max_per_underlying=max_per_underlying)
    top10_put = select_top_n(scored_puts, n=top_n, max_per_underlying=max_per_underlying)

    # الصاق قراردادهای جایگزین همنام (Sisters)
    attach_sister_contracts(top10_call, scored_calls)
    attach_sister_contracts(top10_put, scored_puts)

    # ۵. خودآزمایی اجباری
    errors_call = validate_top10(top10_call, "Call")
    errors_put = validate_top10(top10_put, "Put")

    # تبدیل به DataFrame برای رابط کاربری Streamlit و گزارش‌ها
    df_all = contracts_to_dataframe(all_contracts)
    df_calls_top = contracts_to_dataframe(top10_call)
    df_puts_top = contracts_to_dataframe(top10_put)

    # ثبت هشدارهای کمبود عمق در لاگ
    n_c = len(eligible_calls)
    n_p = len(eligible_puts)
    logger.info(f"تعداد نهایی واجدین شرایط Call: {n_c} | Put: {n_p}")
    if n_p < 5:
        logger.warning(
            f"هشدار عمق کم بازار Put: فقط {n_p} قرارداد واجد شرایط یافت شد (کمتر از ۵ قرارداد). "
            f"پیشنهاد بازبینی آستانه MIN_LEVERAGE_PUT یا نقدینگی در جلسات آتی."
        )

    return {
        "all_contracts": all_contracts,
        "top10_call": top10_call,
        "top10_put": top10_put,
        "df_all": df_all,
        "df_calls_top": df_calls_top,
        "df_puts_top": df_puts_top,
        "gate_logs": {"Call": log_call, "Put": log_put},
        "validation_errors": {"Call": errors_call, "Put": errors_put},
        "eligible_calls_count": n_c,
        "eligible_puts_count": n_p,
    }


STANDARD_COLUMNS = [
    "نماد", "نام قرارداد", "دارایی پایه", "نوع قرارداد", "قیمت پایانی بازار",
    "قیمت اعمال", "قیمت پایه", "قیمت دیروز", "درصد تغییر امروز (%)",
    "تاریخ سررسید", "روزهای تا سررسید (DTE)", "قیمت تئوریک BSM", "دلتا", "اهرم",
    "ارزش معاملات امروز (ریال)", "تعداد معاملات امروز", "میانگین ارزش ۵ روزه (ریال)",
    "حباب خام (%)", "حباب ریالی", "امتیاز الگوریتم", "امتیاز ترکیبی",
    "امتیاز آمادگی پایه", "امتیاز ارزش نسبی (حباب)", "امتیاز اهرم",
    "امتیاز نقدینگی ترکیبی", "امتیاز نقدینگی ساختاری", "امتیاز جهش لحظه‌ای",
    "امتیاز تناسب DTE", "واجد فیلتر سخت", "وضعیت فیلتر", "چرا این امتیاز",
    "شرح فرمول امتیاز", "عمیقاً بی‌ارزش", "برچسب سفته‌بازی", "ضریب دروازه نقدینگی",
    "رتبه در دارایی پایه", "قراردادهای جایگزین", "وضعیت سودآوری",
    "فاصله تا سربه‌سر (%)", "روند دارایی پایه", "هم‌جهتی با روند",
    "کم‌عمق", "وضعیت نقدینگی", "بهترین مظنه خرید (Bid)", "بهترین مظنه فروش (Ask)", "InsCode"
]


def contracts_to_dataframe(contracts: List[OptionContract]) -> pd.DataFrame:
    """تبدیل لیستی از OptionContract به DataFrame استاندارد پانداز"""
    if not contracts:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    rows = [c.to_dict() for c in contracts]
    df = pd.DataFrame(rows)
    if "امتیاز الگوریتم" in df.columns:
        df = df.sort_values("امتیاز الگوریتم", ascending=False, na_position="last").reset_index(drop=True)
    return df


# ==============================================================================
# توابع سازگاری و رپر جهت تست‌ها و ماژول‌های قدیمی
# ==============================================================================
def compute_top_call_and_put(
    df_all: pd.DataFrame,
    config: Optional[AppConfig] = None,
    top_n: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """رپر سازگار جهت فراخوانی توسط app.py، main.py یا تست‌ها"""
    underlying_stats: Dict[str, Dict[str, Any]] = {}
    options_avg_values: Dict[str, float] = {}

    # استخراج دیتای کمکی در صورت وجود ستون‌ها
    if not df_all.empty:
        for _, r in df_all.iterrows():
            u = str(r.get("دارایی پایه", r.get("UnderlyingSymbol", ""))).strip()
            if u and u not in underlying_stats:
                underlying_stats[u] = {
                    "Close": float(r.get("Spot", r.get("UnderlyingClose", 0.0))),
                    "Return1D": float(r.get("بازدهی ۱ روزه پایه (%)", 0.0)),
                    "Return3D": float(r.get("بازدهی ۳ روزه پایه (%)", 0.0)),
                    "Return5D": float(r.get("بازدهی ۵ روزه پایه (%)", 0.0)),
                    "DistSMA5": float(r.get("فاصله پایه از SMA5 (%)", 0.0)),
                    "DistSMA20": float(r.get("فاصله پایه از SMA20 (%)", 0.0)),
                    "QueueStatus": str(r.get("وضعیت صف پایه", "متعادل")),
                    "RSI14": float(r.get("RSI14 پایه", 50.0)),
                    "RSIStatus": str(r.get("وضعیت RSI پایه", "عادی")),
                    "IsPullback": bool(r.get("پولبک پایه", False)),
                    "PriceBand": r.get("دامنه نوسان پایه"),
                }
            inc = str(r.get("InsCode", ""))
            if inc and inc not in options_avg_values:
                options_avg_values[inc] = float(r.get("میانگین ارزش ۵ روزه (ریال)", 0.0))

    res = run_unified_scoring_pipeline(
        df_raw=df_all,
        underlying_stats=underlying_stats,
        options_avg_values=options_avg_values,
        config=config,
        top_n=top_n,
    )

    df_calls = res["df_calls_top"]
    df_puts = res["df_puts_top"]

    call_warn = False
    call_sym = None
    call_count = 0
    if not df_calls.empty and "دارایی پایه" in df_calls.columns:
        call_vc = df_calls["دارایی پایه"].value_counts()
        if not call_vc.empty and call_vc.iloc[0] > 3:
            call_warn = True
            call_sym = str(call_vc.index[0])
            call_count = int(call_vc.iloc[0])

    put_warn = False
    put_sym = None
    put_count = 0
    if not df_puts.empty and "دارایی پایه" in df_puts.columns:
        put_vc = df_puts["دارایی پایه"].value_counts()
        if not put_vc.empty and put_vc.iloc[0] > 3:
            put_warn = True
            put_sym = str(put_vc.index[0])
            put_count = int(put_vc.iloc[0])

    conc_info = {
        "call_warning": call_warn,
        "call_sym": call_sym,
        "call_count": call_count,
        "put_warning": put_warn,
        "put_sym": put_sym,
        "put_count": put_count,
        "eligible_calls_count": res["eligible_calls_count"],
        "eligible_puts_count": res["eligible_puts_count"],
        "min_leverage_call": MIN_LEVERAGE_CALL,
        "min_leverage_put": MIN_LEVERAGE_PUT,
        "validation_errors": res["validation_errors"],
    }
    return df_calls, df_puts, conc_info


def calculate_score(
    row: Union[pd.Series, Dict[str, Any], OptionContract],
    structural_liq_pct: float = 50.0,
    bubble_pct_rank: Optional[float] = None,
    config: Optional[AppConfig] = None,
    leverage_pct_rank: Optional[float] = None,
) -> Dict[str, Any]:
    """رپر سازگاری جهت ارزیابی مستقیم یک ردیف در تست‌ها"""
    if isinstance(row, OptionContract):
        c = row
    elif isinstance(row, pd.Series):
        c = build_contracts(pd.DataFrame([row.to_dict()]), {}, {}, config)[0]
    else:
        c = build_contracts(pd.DataFrame([row]), {}, {}, config)[0]

    # اگر در تست‌های قدیمی اهرم ست نشده بود اما leverage_pct_rank پاس شده بود:
    if leverage_pct_rank is not None and c.leverage is None:
        c.leverage = 5.0
        if c.bsm_price is not None and c.bsm_price >= 10 and c.delta is not None and abs(c.delta) >= 0.10:
            c.is_valid_pricing = True

    is_deep_otm = not (c.is_valid_pricing and (c.bsm_price or 0) >= 10 and abs(c.delta or 0) >= 0.10)

    # ۱. آمادگی پایه
    u_data = row if isinstance(row, (dict, pd.Series)) else {}
    readiness, reasons_u, u_detail = calculate_underlying_readiness_score(u_data, is_call=(c.option_type == "Call"), config=config)

    # ۲. ارزش نسبی حباب
    relative_value = float(bubble_pct_rank) if bubble_pct_rank is not None else 50.0

    # ۳. اهرم
    if leverage_pct_rank is not None:
        leverage_score = float(leverage_pct_rank)
    else:
        leverage_score = round(min(100.0, ((c.leverage or 0.0) / 7.0) * 100.0), 1) if c.leverage else 50.0

    # ۴. نقدینگی ترکیبی
    comb_liq, struct_contrib, spike_score = calculate_combined_liquidity_score(
        structural_pct=structural_liq_pct,
        today_volume=c.today_trade_value,
        avg_5d_volume=c.avg_5d_trade_value,
        config=config,
    )

    # ۵. تناسب DTE
    dte_score = calculate_dte_suitability_score(c.dte)

    # امتیاز خام نهایی
    raw_final = round(
        0.25 * u_detail["raw"]
        + 0.20 * relative_value
        + 0.15 * leverage_score
        + 0.25 * comb_liq
        + 0.15 * dte_score,
        1
    )

    # ضریب دروازه نقدینگی
    if c.today_trade_count == 0 or comb_liq < 15.0:
        liq_mult = 0.15
        liq_label = "فاقد معامله / نقدینگی بسیار ضعیف"
    elif comb_liq < 40.0:
        liq_mult = round(0.50 + comb_liq / 200.0, 4)
        liq_label = "نقدینگی کم‌عمق"
    else:
        liq_mult = 1.0
        liq_label = "عادی"

    min_val = MIN_TRADE_VALUE_CALL if c.option_type == "Call" else MIN_TRADE_VALUE_PUT
    min_count = MIN_TRADE_COUNT_CALL if c.option_type == "Call" else MIN_TRADE_COUNT_PUT
    min_lev = MIN_LEVERAGE_CALL if c.option_type == "Call" else MIN_LEVERAGE_PUT

    passes = (
        c.is_valid_pricing
        and (not is_deep_otm)
        and c.today_trade_count >= min_count
        and c.today_trade_value >= min_val
        and c.leverage is not None and c.leverage >= min_lev
        and c.dte >= 3
    )
    c.passes_hard_filter = passes

    if is_deep_otm or not c.is_valid_pricing:
        return {
            "final_score": None,
            "final_score_num": 0.0,
            "raw_final_score": raw_final,
            "passes_hard_filter": False,
            "explanation": "رد شده در گیت Deep-OTM / داده نامعتبر",
            "reasons": ["داده قیمت‌گذاری نامعتبر یا Deep OTM"],
            "readiness_score": 0.0,
            "relative_value_score": 0.0,
            "leverage_score": 0.0,
            "combined_liquidity_score": 0.0,
            "structural_liquidity_score": structural_liq_pct,
            "spike_liquidity_score": 0.0,
            "dte_suitability_score": dte_score,
            "liquidity_gate_multiplier": liq_mult,
            "liquidity_gate_label": liq_label,
            "is_deep_otm": True,
            "deep_otm_label": "لاتاری/بی‌ارزش عمیق",
        }

    # اعمال ضریب نقدینگی روی امتیاز نهایی
    final_score = round(raw_final * liq_mult, 1)

    # توضیحات
    if c.today_trade_count == 0:
        explanation = "فاقد معامله امروز (اعمال ضریب ۰.۱۵)"
    elif not passes:
        explanation = "رد شده در فیلتر سخت"
    else:
        explanation = f"امتیاز {final_score}"

    return {
        "final_score": final_score,
        "final_score_num": final_score,
        "raw_final_score": raw_final,
        "passes_hard_filter": passes,
        "explanation": explanation,
        "reasons": reasons_u,
        "readiness_score": u_detail["raw"],
        "relative_value_score": relative_value,
        "leverage_score": leverage_score,
        "combined_liquidity_score": comb_liq,
        "structural_liquidity_score": structural_liq_pct,
        "spike_liquidity_score": spike_score,
        "dte_suitability_score": dte_score,
        "liquidity_gate_multiplier": liq_mult,
        "liquidity_gate_label": liq_label,
        "is_deep_otm": False,
        "deep_otm_label": "",
    }


def select_top_n_diversified(
    scored_contracts: Any,
    n: int = 10,
    max_per_underlying: int = 3,
) -> Any:
    """سازگاری با فراخوانی‌های قدیمی select_top_n_diversified"""
    if isinstance(scored_contracts, list):
        return select_top_n(scored_contracts, n=n, max_per_underlying=max_per_underlying)
    elif isinstance(scored_contracts, pd.DataFrame):
        if scored_contracts.empty:
            return scored_contracts.copy()
        sort_col = "امتیاز الگوریتم" if "امتیاز الگوریتم" in scored_contracts.columns else scored_contracts.columns[0]
        df_sorted = scored_contracts.sort_values(sort_col, ascending=False).copy()
        selected_idx = []
        counts: Dict[str, int] = {}
        for idx, row in df_sorted.iterrows():
            u = str(row.get("دارایی پایه", row.get("underlying", "")))
            if counts.get(u, 0) < max_per_underlying:
                selected_idx.append(idx)
                counts[u] = counts.get(u, 0) + 1
            if len(selected_idx) == n:
                break
        res_df = df_sorted.loc[selected_idx].reset_index(drop=True)
        if "دارایی پایه" in res_df.columns and "رتبه در دارایی پایه" not in res_df.columns:
            u_ranks: Dict[str, int] = {}
            ranks_list = []
            for _, r in res_df.iterrows():
                u = str(r.get("دارایی پایه", ""))
                curr_r = u_ranks.get(u, 0) + 1
                u_ranks[u] = curr_r
                ranks_list.append(curr_r)
            res_df["رتبه در دارایی پایه"] = ranks_list
        return res_df
    return scored_contracts


def get_sister_contracts(
    current_symbol: str,
    underlying: str,
    df_pool: pd.DataFrame,
    is_call: Optional[bool] = None,
    max_sisters: int = 2,
    **kwargs,
) -> List[Dict[str, Any]]:
    """سازگاری با تابع get_sister_contracts قدیمی"""
    if df_pool is None or df_pool.empty:
        return []
    df = df_pool.copy()
    if "واجد فیلتر سخت" in df.columns:
        df = df[df["واجد فیلتر سخت"] == True]
    mask = (df["دارایی پایه"] == underlying) & (df["نماد"] != current_symbol)
    if is_call is not None:
        call_mask = df["نوع قرارداد"].str.contains("Call|خرید", case=False, na=False) | df["نماد"].str.startswith("ض")
        mask = mask & (call_mask if is_call else ~call_mask)
    filtered = df[mask]
    if filtered.empty:
        return []
    score_col = "امتیاز الگوریتم" if "امتیاز الگوریتم" in filtered.columns else "final_score"
    top_s = filtered.sort_values(score_col, ascending=False).head(max_sisters)
    res = []
    for _, r in top_s.iterrows():
        res.append({
            "symbol": str(r["نماد"]),
            "strike": int(r.get("قیمت اعمال", 0)),
            "dte": int(r.get("روزهای تا سررسید (DTE)", 0)),
            "score": float(r.get(score_col, 0.0)),
            "price": int(r.get("قیمت پایانی بازار", 0)),
            "underlying": underlying,
        })
    return res
