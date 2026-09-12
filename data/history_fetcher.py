"""
ماژول دریافت تاریخچه قیمت، ارزش معاملات و صف‌های خرید/فروش برای دارایی‌های پایه و قراردادهای اختیار
"""

import io
import math
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import requests

from data.normalizer import normalize_fa, symbols_match
from analytics.underlying_momentum import determine_momentum_label

logger = logging.getLogger("OptionScanner.HistoryFetcher")

# کش سراسری حافظه برای میانگین ۵ روزه ارزش معاملات قراردادها (طول عمر: ۴ ساعت)
_OPTION_5D_CACHE: Dict[str, Tuple[float, float]] = {}
CACHE_TTL_SECONDS = 4 * 3600  # ۴ ساعت


class HistoryFetcher:
    """کلاس استخراج سابقه قیمتی، محاسبه شاخص‌های تکنیکال دارایی پایه و میانگین ۵ روزه معاملات با عملکرد فوق‌سریع"""

    def __init__(self, timeout: int = 8, max_workers: int = 25):
        self.timeout = timeout
        self.max_workers = max_workers
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        # بازیافت اتصالات TCP و بهینه‌سازی شبکه با Connection Pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers,
            pool_maxsize=max_workers,
            max_retries=1,
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(self.headers)

    def get_underlying_history_and_stats(
        self,
        symbol: str,
        limit: int = 30,
        pre_info: Optional[Dict[str, Any]] = None,
        queue_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        دریافت تاریخچه قیمت سهام پایه و محاسبه:
        - درصدهای تغییر ۱، ۳، ۵ و ۱۰ روزه
        - میانگین متحرک ۵ و ۲۰ روزه (SMA5, SMA20) و فاصله از آنها
        - نوسان‌پذیری تاریخی سالانه (Realized Volatility)
        - وضعیت صف خرید/فروش (استفاده از کش دسته‌ای برای جلوگیری از درخواست‌های تکراری)
        """
        import algotik_tse as alt

        norm_sym = normalize_fa(symbol)
        stats = {
            "Symbol": symbol,
            "NormalizedSymbol": norm_sym,
            "Close": 0.0,
            "Return1D": 0.0,
            "Return3D": 0.0,
            "Return5D": 0.0,
            "Return10D": 0.0,
            "SMA5": 0.0,
            "SMA20": 0.0,
            "DistSMA5": 0.0,
            "DistSMA20": 0.0,
            "RSI14": 50.0,
            "PrevRSI14": 50.0,
            "RSIStatus": "عادی",
            "IsPullback": False,
            "PullbackDesc": "ندارد",
            "IsNearSMA20": False,
            "RealizedVol": 0.35,  # پیش‌فرض ۳۵٪ در صورت ناکافی بودن دیتا
            "QueueStatus": "متعادل",
            "QueueVolumeRatio": 1.0,
            "HistoryDays": 0,
            "PriceBand": 0.03,
            "PriceBandSource": "Default",
            "IsLeveraged": False,
            "Threshold3D": 8.35,
            "Threshold1D": 2.70,
        }

        try:
            df_hist = alt.get_history(symbol=symbol, limit=limit, include_today=False, progress=False)
            if df_hist is None or df_hist.empty or "Close" not in df_hist.columns:
                logger.warning(f"تاریخچه قیمت برای نماد پایه {symbol} یافت نشد.")
                if pre_info and pre_info.get("Close", 0.0) > 0:
                    stats["Close"] = float(pre_info["Close"])
                return stats

            closes = df_hist["Close"].astype(float).values
            n = len(closes)
            stats["HistoryDays"] = n
            if n > 0:
                current_close = float(closes[-1])
                stats["Close"] = current_close

                # بازدهی‌های دوره‌ای
                if n >= 2 and closes[-2] > 0:
                    stats["Return1D"] = round(((current_close - closes[-2]) / closes[-2]) * 100.0, 2)
                if n >= 4 and closes[-4] > 0:
                    stats["Return3D"] = round(((current_close - closes[-4]) / closes[-4]) * 100.0, 2)
                if n >= 6 and closes[-6] > 0:
                    stats["Return5D"] = round(((current_close - closes[-6]) / closes[-6]) * 100.0, 2)
                if n >= 11 and closes[-11] > 0:
                    stats["Return10D"] = round(((current_close - closes[-11]) / closes[-11]) * 100.0, 2)

                # میانگین متحرک SMA5
                if n >= 5:
                    sma5 = float(np.mean(closes[-5:]))
                    stats["SMA5"] = round(sma5, 1)
                    if sma5 > 0:
                        dist5 = ((current_close - sma5) / sma5) * 100.0
                        stats["DistSMA5"] = round(dist5, 2)

                # میانگین متحرک SMA20
                if n >= 20:
                    sma20 = float(np.mean(closes[-20:]))
                    stats["SMA20"] = round(sma20, 1)
                    if sma20 > 0:
                        dist20 = ((current_close - sma20) / sma20) * 100.0
                        stats["DistSMA20"] = round(dist20, 2)
                        stats["IsNearSMA20"] = abs(dist20) <= 2.5
                elif n >= 5:
                    sma20 = float(np.mean(closes))
                    stats["SMA20"] = round(sma20, 1)
                    if sma20 > 0:
                        dist20 = ((current_close - sma20) / sma20) * 100.0
                        stats["DistSMA20"] = round(dist20, 2)
                        stats["IsNearSMA20"] = abs(dist20) <= 2.5

                # محاسبه شاخص قدرت نسبی RSI (دوره ۱۴ روزه استاندارد با هموارسازی وایلدر)
                if n >= 15:
                    deltas = np.diff(closes)
                    period = 14
                    seed = deltas[:period]
                    up = float(seed[seed >= 0].sum()) / period
                    down = float(-seed[seed < 0].sum()) / period
                    rs = up / down if down != 0 else 1000.0
                    rsi_series = [100.0 - (100.0 / (1.0 + rs))]

                    for i in range(period, len(deltas)):
                        delta = deltas[i]
                        upval = delta if delta > 0 else 0.0
                        downval = -delta if delta < 0 else 0.0
                        up = (up * (period - 1) + upval) / period
                        down = (down * (period - 1) + downval) / period
                        rs = up / down if down != 0 else 1000.0
                        rsi_series.append(100.0 - (100.0 / (1.0 + rs)))

                    curr_rsi = round(float(rsi_series[-1]), 1)
                    prev_rsi = round(float(rsi_series[-2]), 1) if len(rsi_series) >= 2 else curr_rsi
                    stats["RSI14"] = curr_rsi
                    stats["PrevRSI14"] = prev_rsi

                    # تعیین برچسب وضعیت RSI
                    if curr_rsi >= 70:
                        stats["RSIStatus"] = "اشباع خرید (>70)"
                    elif curr_rsi <= 30:
                        stats["RSIStatus"] = "اشباع فروش (<30)"
                    elif (prev_rsi <= 35 and curr_rsi > prev_rsi) or (30 <= curr_rsi <= 45 and curr_rsi > prev_rsi):
                        stats["RSIStatus"] = "خروج از اشباع فروش (چرخش مثبت)"
                    else:
                        stats["RSIStatus"] = "عادی (خنثی)"

                # تشخیص الگوی پولبک (افت ۲ تا ۵ روزه پس از صعود مستند)
                ret_10 = stats.get("Return10D", 0.0)
                ret_5 = stats.get("Return5D", 0.0)
                ret_3 = stats.get("Return3D", 0.0)
                ret_1 = stats.get("Return1D", 0.0)

                # اگر روند میان‌مدت (۱۰ یا ۵ روزه) مثبت بوده ولی در ۳ روز اخیر افت خفیف/اصلاح داشته
                if (ret_10 > 2.0 or ret_5 > 1.5) and (ret_3 < 0.5 or (n >= 4 and closes[-1] < closes[-3])):
                    stats["IsPullback"] = True
                    stats["PullbackDesc"] = "پولبک سالم پس از صعود"
                elif ret_10 < -5.0 and ret_5 < -3.0:
                    stats["IsPullback"] = False
                    stats["PullbackDesc"] = "روند نزولی ممتد (فاقد پولبک)"
                else:
                    stats["IsPullback"] = False
                    stats["PullbackDesc"] = "روند نوسانی / عادی"

                # محاسبه نوسان‌پذیری تاریخی تحقق‌یافته (Realized Volatility سالانه)
                if n >= 10:
                    log_returns = np.diff(np.log(closes))
                    daily_std = float(np.std(log_returns, ddof=1))
                    annualized_vol = daily_std * math.sqrt(240)  # ۲۴۰ روز کاری در بورس تهران
                    stats["RealizedVol"] = round(max(0.15, min(annualized_vol, 1.50)), 4)

            # بررسی وضعیت صف خرید/فروش (استفاده از دیتای پیش‌پردازش شده در حافظه RAM بدون درخواست مکرر شبکه)
            if queue_status is not None:
                stats["QueueStatus"] = queue_status
            else:
                try:
                    ob = alt.get_order_book(symbol=symbol)
                    if ob is not None and not ob.empty and "BidVolume" in ob.columns and "AskVolume" in ob.columns:
                        tot_bid_vol = float(ob["BidVolume"].sum())
                        tot_ask_vol = float(ob["AskVolume"].sum())
                        if tot_bid_vol > 0 and tot_ask_vol == 0:
                            stats["QueueStatus"] = "صف خرید"
                        elif tot_ask_vol > 0 and tot_bid_vol == 0:
                            stats["QueueStatus"] = "صف فروش"
                        elif tot_bid_vol > tot_ask_vol * 3.0:
                            stats["QueueStatus"] = "برتری تقاضا (صف خرید نسبی)"
                        elif tot_ask_vol > tot_bid_vol * 3.0:
                            stats["QueueStatus"] = "برتری عرضه (صف فروش نسبی)"
                        else:
                            stats["QueueStatus"] = "متعادل"
                except Exception as q_err:
                    logger.debug(f"عدم امکان خواندن صف برای نماد {symbol}: {q_err}")

            # استخراج پویای دامنه نوسان مجاز (Price Band) از دیده‌بان دسته‌ای (بدون درخواست وب اضافی)
            band_found = False
            if pre_info:
                mx_p = float(pre_info.get("MaxAllowed", 0.0) or 0.0)
                mn_p = float(pre_info.get("MinAllowed", 0.0) or 0.0)
                if mx_p > mn_p > 0:
                    mid_p = (mx_p + mn_p) / 2.0
                    detected_band = round((mx_p - mid_p) / mid_p, 4)
                    if 0.005 <= detected_band <= 0.20:
                        stats["PriceBand"] = detected_band
                        stats["PriceBandSource"] = "TSETMC Dynamic (Batch)"
                        stats["IsLeveraged"] = detected_band >= 0.038
                        band_found = True

            if not band_found:
                try:
                    inf = alt.get_info(symbol=symbol)
                    if inf is not None and not inf.empty:
                        if "staticThreshold_psGelStaMax" in inf.index and "staticThreshold_psGelStaMin" in inf.index:
                            mx_p = float(inf.loc["staticThreshold_psGelStaMax", "value"])
                            mn_p = float(inf.loc["staticThreshold_psGelStaMin", "value"])
                            if mx_p > mn_p > 0:
                                mid_p = (mx_p + mn_p) / 2.0
                                detected_band = round((mx_p - mid_p) / mid_p, 4)
                                if 0.005 <= detected_band <= 0.20:
                                    stats["PriceBand"] = detected_band
                                    stats["PriceBandSource"] = "TSETMC Dynamic"
                                    stats["IsLeveraged"] = detected_band >= 0.038
                except Exception as p_err:
                    logger.debug(f"عدم امکان خواندن دامنه نوسان پویا برای نماد {symbol}: {p_err}")

        except Exception as e:
            logger.error(f"خطا در استخراج آمار دارایی پایه {symbol}: {e}")

        # در صورت عدم استخراج پویا، از فالبک نمادهای اهرمی / عادی استفاده می‌شود
        if stats.get("PriceBandSource") == "Default":
            is_lev = any(symbols_match(norm_sym, s) for s in ["اهرم", "توان", "موج", "اطلس", "جهش", "شتاب", "نارنج", "بیدار"])
            stats["PriceBand"] = 0.04 if is_lev else 0.03
            stats["PriceBandSource"] = "Fallback (Leveraged)" if is_lev else "Fallback (Regular)"
            stats["IsLeveraged"] = is_lev

        band = float(stats["PriceBand"])
        max_3day_move = ((1.0 + band) ** 3) - 1.0
        stats["Threshold3D"] = round(0.90 * max_3day_move * 100.0, 2)
        stats["Threshold1D"] = round(0.90 * band * 100.0, 2)

        stats["MomentumLabel"] = determine_momentum_label(stats)
        return stats

    def fetch_all_underlying_stats(
        self,
        symbols: List[str],
        market_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """دریافت همزمان آمار تمام دارایی‌های پایه هدف با واکشی تک‌باره دیده‌بان کل بازار"""
        import algotik_tse as alt

        logger.info(f"شروع واکشی پرسرعت آمار {len(symbols)} نماد دارایی پایه...")
        snap = market_snapshot
        if snap is None:
            try:
                snap = alt.market_watch()
            except Exception as e:
                logger.warning(f"عدم امکان دریافت یک‌باره market_watch ({e})، از فالبک عادی استفاده می‌شود.")
                snap = None

        stocks_df = snap.get("stocks") if snap else None
        order_book_df = snap.get("order_book") if snap else None

        # پیش‌پردازش دامنه‌های مجاز و مشخصات پایه در حافظه RAM
        preprocessed_info: Dict[str, Dict[str, Any]] = {}
        if stocks_df is not None and not stocks_df.empty:
            stocks_df = stocks_df.copy()
            stocks_df["NormSym"] = stocks_df["Symbol"].astype(str).apply(normalize_fa)
            for _, s_row in stocks_df.iterrows():
                ns = s_row["NormSym"]
                preprocessed_info[ns] = {
                    "InsCode": str(s_row.get("InsCode", "")),
                    "MaxAllowed": float(s_row.get("MaxAllowed", 0.0) or 0.0),
                    "MinAllowed": float(s_row.get("MinAllowed", 0.0) or 0.0),
                    "Close": float(s_row.get("Close", 0.0) or 0.0),
                    "Last": float(s_row.get("Last", 0.0) or 0.0),
                }

        # پیش‌پردازش وضعیت صف‌ها از دفتر سفارشات در حافظه RAM
        queue_info: Dict[str, str] = {}
        if order_book_df is not None and not order_book_df.empty:
            try:
                ob_grouped = order_book_df.groupby("InsCode")[["BidVolume", "AskVolume"]].sum()
                for inscode_val, q_row in ob_grouped.iterrows():
                    icode_str = str(inscode_val)
                    b_vol = float(q_row["BidVolume"])
                    a_vol = float(q_row["AskVolume"])
                    if b_vol > 0 and a_vol == 0:
                        q_stat = "صف خرید"
                    elif a_vol > 0 and b_vol == 0:
                        q_stat = "صف فروش"
                    elif b_vol > a_vol * 3.0:
                        q_stat = "برتری تقاضا (صف خرید نسبی)"
                    elif a_vol > b_vol * 3.0:
                        q_stat = "برتری عرضه (صف فروش نسبی)"
                    else:
                        q_stat = "متعادل"
                    queue_info[icode_str] = q_stat
            except Exception as ob_err:
                logger.debug(f"خطا در پردازش دسته‌ای دفتر سفارشات: {ob_err}")

        results = {}
        with ThreadPoolExecutor(max_workers=min(len(symbols), self.max_workers)) as executor:
            future_to_sym = {}
            for sym in symbols:
                norm_s = normalize_fa(sym)
                p_info = preprocessed_info.get(norm_s)
                icode = p_info.get("InsCode", "") if p_info else ""
                q_stat = queue_info.get(icode) if icode else None
                future = executor.submit(
                    self.get_underlying_history_and_stats,
                    sym,
                    30,
                    p_info,
                    q_stat,
                )
                future_to_sym[future] = sym

            for future in as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    stats = future.result()
                    results[sym] = stats
                    results[stats["NormalizedSymbol"]] = stats
                except Exception as exc:
                    logger.error(f"استخراج نماد {sym} با خطا مواجه شد: {exc}")

        logger.info(f"آمار تکنیکال {len(results)} دارایی پایه با موفقیت آماده شد.")
        return results

    def _fetch_single_option_avg_value(self, inscode: str, lookback_days: int = 5) -> float:
        """استخراج میانگین ارزش معامله ۵ روز اخیر با اولویت CDN جدید تستی TSETMC و فالبک هوشمند"""
        if not inscode or str(inscode) == "0":
            return 0.0

        import time
        now = time.time()
        if inscode in _OPTION_5D_CACHE:
            val, ts = _OPTION_5D_CACHE[inscode]
            if now - ts < CACHE_TTL_SECONDS:
                return val

        # ۱. تلاش اول: اندپوینت پرسرعت و فوق‌سبک CDN اختصاصی TSETMC (حجم ۱.۶ کیلوبایت در برابر ۴۵۰ کیلوبایت)
        cdn_url = f"https://cdn.tsetmc.com/api/ClosingPrice/GetClosingPriceDailyList/{inscode}/{lookback_days}"
        try:
            resp = self.session.get(cdn_url, timeout=2.5)
            if resp.status_code == 200:
                payload = resp.json()
                items = payload.get("closingPriceDaily", []) if isinstance(payload, dict) else []
                vals = [float(x["qTotCap"]) for x in items if isinstance(x, dict) and "qTotCap" in x and x["qTotCap"] is not None]
                if vals:
                    res = float(np.mean(vals))
                    _OPTION_5D_CACHE[inscode] = (res, now)
                    return res
        except Exception as cdn_err:
            logger.debug(f"عدم امکان دریافت داده از CDN برای نماد {inscode}: {cdn_err}")

        # ۲. تلاش دوم (فالبک): استفاده از اندپوینت قدیمی متنی Export-txt
        url = f"https://old.tsetmc.com/tsev2/data/Export-txt.aspx?t=i&a=1&b=0&i={inscode}"
        try:
            resp = self.session.get(url, timeout=3.5)
            if resp.status_code == 200 and resp.text:
                lines = resp.text.strip().splitlines()
                if len(lines) > 1:
                    header = [h.strip("<>").upper() for h in lines[0].split(",")]
                    val_idx = header.index("VALUE") if "VALUE" in header else 6

                    daily_values = []
                    data_rows = lines[1:]
                    for row in data_rows[-lookback_days:]:
                        parts = row.split(",")
                        if len(parts) > val_idx:
                            try:
                                v = float(parts[val_idx])
                                daily_values.append(v)
                            except ValueError:
                                pass

                    if daily_values:
                        res = float(np.mean(daily_values))
                        _OPTION_5D_CACHE[inscode] = (res, now)
                        return res
        except Exception as e:
            logger.debug(f"خطا در دریافت تاریخچه ارزش معامله نماد با InsCode {inscode}: {e}")

        _OPTION_5D_CACHE[inscode] = (0.0, now)
        return 0.0

    def fetch_options_5d_avg_values(
        self,
        inscodes: List[str],
        lookback_days: int = 5,
        active_only_inscodes: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        استخراج موازی میانگین ۵ روزه ارزش معاملات برای تمام نمادهای آپشن با کش سراسری و ThreadPool.
        """
        unique_inscodes = list(set(str(c) for c in inscodes if c and str(c) != "0"))

        # بررسی موارد موجود در کش
        avg_values: Dict[str, float] = {}
        missing_inscodes = []
        import time
        now = time.time()
        for c in unique_inscodes:
            if c in _OPTION_5D_CACHE and (now - _OPTION_5D_CACHE[c][1] < CACHE_TTL_SECONDS):
                avg_values[c] = _OPTION_5D_CACHE[c][0]
            else:
                missing_inscodes.append(c)

        if missing_inscodes:
            # اگر فهرست نمادهای فعال مشخص باشد، فقط موارد فعال را از شبکه می‌گیریم
            to_fetch = missing_inscodes
            if active_only_inscodes is not None:
                active_set = set(str(c) for c in active_only_inscodes)
                to_fetch = [c for c in missing_inscodes if c in active_set]
                # قراردادهای غیرفعال بدون معامله مقدار صفر می‌گیرند
                for c in missing_inscodes:
                    if c not in active_set:
                        avg_values[c] = 0.0
                        _OPTION_5D_CACHE[c] = (0.0, now)

            if to_fetch:
                logger.info(f"در حال استخراج موازی میانگین ۵ روزه برای {len(to_fetch)} نماد فعال با CDN اختصاصی...")
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_code = {
                        executor.submit(self._fetch_single_option_avg_value, code, lookback_days): code
                        for code in to_fetch
                    }
                    for future in as_completed(future_to_code):
                        code = future_to_code[future]
                        try:
                            avg_values[code] = future.result()
                        except Exception:
                            avg_values[code] = 0.0

        logger.info(f"میانگین ۵ روزه ارزش معاملات برای {len(avg_values)} نماد با موفقیت آماده شد.")
        return avg_values
