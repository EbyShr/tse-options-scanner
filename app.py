"""
وب‌اپلیکیشن اسکنر و رتبه‌بند روزانه بازار آپشن بورس تهران (TSETMC)
طراحی پریمیوم: الگوریتم امتیازدهی یکپارچه، تفکیک ۱۰ برتر Call و Put، چارت‌های راداری Plotly،
پالت دارک/لایت، مهار Deep OTM، سقف ضد-Chasing و سرعت پردازش موازی.
"""

import os
import sys
import time
import pickle
import logging
from datetime import datetime, timedelta
import jdatetime
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

# تنظیم خروجی کنسول به UTF-8
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import load_config, AppConfig
from data.market_fetcher import MarketFetcher
from data.history_fetcher import HistoryFetcher
from analytics.metrics import process_options_dataframe
from analytics.unified_scoring import (
    compute_top_call_and_put,
    calculate_score,
    get_sister_contracts,
    validate_top10,
    MIN_LEVERAGE_CALL,
    MIN_LEVERAGE_PUT,
)
from reports.csv_exporter import create_top_choices_overview_df, export_top_choices_csv

CACHE_FILE = os.path.join("cache", "market_data_cache.pkl")

# ==============================================================================
# ۱. پیکربندی صفحه و استایل راست‌چین (RTL)
# ==============================================================================
st.set_page_config(
    page_title="اسکنر و رتبه‌بند آپشن بورس تهران",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# ۲. لایه کشینگ هوشمند و پایپ‌لاین موازی
# ==============================================================================
def load_disk_cache():
    """بازیابی داده‌های اسکن از فایل کش محلی"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.sidebar.error(f"خطا در خواندن فایل کش: {e}")
    return None


def save_disk_cache(data_dict):
    """ذخیره داده‌های اسکن در فایل کش محلی"""
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(data_dict, f)
    except Exception as e:
        st.sidebar.warning(f"عدم امکان ذخیره کش در دیسک: {e}")


@st.cache_data(ttl=25, show_spinner=False)
def fetch_raw_market_snapshot(target_symbols_tuple):
    """دریافت اسنپ‌شات اتمیک بازار با کش ۲۵ ثانیه‌ای جهت جلوگیری از درخواست‌های مکرر"""
    mf = MarketFetcher(request_timeout=10)
    return mf.fetch_options_snapshot(list(target_symbols_tuple))


@st.cache_data(ttl=300, show_spinner=False)
def fetch_cached_underlying_stats(target_symbols_tuple, request_timeout=8):
    """استخراج تاریخچه، شاخص‌های تکنیکال، دیتای ۳‌روزه و سقف نوسان دارایی‌های پایه با کش ۵ دقیقه‌ای (۳۰۰ ثانیه)"""
    history_fetcher = HistoryFetcher(timeout=request_timeout, max_workers=20)
    return history_fetcher.fetch_all_underlying_stats(list(target_symbols_tuple))


def execute_pipeline(config: AppConfig, progress_container=None):
    """
    اجرای کامل پایپ‌لاین با موازیسازی حداکثری و نمایش پیشرفت تدریجی
    """
    failed_symbols = []
    target_symbols = tuple(config.target_underlying_symbols)

    if progress_container:
        p_bar = progress_container.progress(0.1, text="۱/۴: در حال دریافت اسنپ‌شات اتمیک بازار آپشن...")
    else:
        p_bar = None

    # ۱. استخراج اتمیک زنجیره بازار
    df_raw = fetch_raw_market_snapshot(target_symbols)
    if df_raw.empty:
        raise ConnectionError("دریافت اطلاعات از سرورهای TSETMC ناموفق بود.")

    if p_bar:
        p_bar.progress(0.4, text=f"۲/۴: استخراج موازی شاخص‌های {len(target_symbols)} دارایی پایه (کش ۳۰۰ ثانیه‌ای)...")

    # ۲. استخراج موازی دارایی‌های پایه (کش‌شده ۵ دقیقه‌ای)
    history_fetcher = HistoryFetcher(timeout=config.network.request_timeout, max_workers=20)
    underlying_stats = fetch_cached_underlying_stats(
        target_symbols, request_timeout=config.network.request_timeout
    )

    if p_bar:
        p_bar.progress(0.7, text="۳/۴: پردازش میانگین ۵ روزه ارزش معاملات و نقدینگی...")

    # ۳. استخراج میانگین ۵ روزه ارزش معاملات قراردادهای فعال
    active_inscodes = df_raw[df_raw["Value"] > 0]["InsCode"].dropna().astype(str).tolist()
    all_inscodes = df_raw["InsCode"].dropna().astype(str).tolist()
    options_avg = history_fetcher.fetch_options_5d_avg_values(
        all_inscodes,
        lookback_days=config.liquidity_filter.lookback_days_avg,
        active_only_inscodes=active_inscodes,
    )

    if p_bar:
        p_bar.progress(0.9, text="۴/۴: ارزش‌گذاری بلک-شولز، حباب، اهرم و الگوریتم واحد...")

    # ۴. پردازش نهایی متریک‌ها
    df_processed = process_options_dataframe(
        df_options=df_raw,
        underlying_stats=underlying_stats,
        options_avg_values=options_avg,
        config=config,
    )

    now_dt = datetime.now()
    now_jalali = jdatetime.datetime.now().strftime("%Y-%m-%d ساعت %H:%M:%S")

    package = {
        "df_processed": df_processed,
        "underlying_stats": underlying_stats,
        "failed_symbols": failed_symbols,
        "timestamp_jalali": now_jalali,
        "timestamp_unix": now_dt.timestamp(),
        "total_count": len(df_processed),
    }

    save_disk_cache(package)
    if p_bar:
        p_bar.progress(1.0, text="تکمیل شد!")
        time.sleep(0.3)
        progress_container.empty()

    return package


# ==============================================================================
# ۳. بارگذاری تنظیمات و راه‌اندازی سشن
# ==============================================================================
config = load_config("config.yaml")

if "market_data" not in st.session_state:
    cached = load_disk_cache()
    if cached is not None:
        st.session_state["market_data"] = cached
        st.session_state["is_cached_fallback"] = False
    else:
        init_container = st.empty()
        try:
            st.session_state["market_data"] = execute_pipeline(config, init_container)
            st.session_state["is_cached_fallback"] = False
        except Exception as e:
            st.error(f"خطا در استخراج اولیه داده‌ها: {e}")
            st.stop()


# ==============================================================================
# ۴. سایدبار: تم ظاهری، کنترل رفرش خودکار و اطلاعات بازار
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ کنترل پنل اسکنر")

    # انتخاب قالب رنگی (Dark / Light Theme)
    theme_choice = st.radio(
        "پالت بصری:",
        options=["تیره موسساتی (Dark Navy)", "روشن مدرن (Light)"],
        index=0,
        horizontal=True,
    )
    is_dark = "Dark" in theme_choice

    st.markdown("---")

    # رفرش دستی
    if st.button("🔄 به‌روزرسانی زنده داده‌ها (Refresh)", use_container_width=True):
        fetch_raw_market_snapshot.clear()
        fetch_cached_underlying_stats.clear()
        refresh_box = st.empty()
        try:
            st.session_state["market_data"] = execute_pipeline(config, refresh_box)
            st.session_state["is_cached_fallback"] = False
            st.success("داده‌ها با موفقیت به‌روزرسانی شدند!")
            st.rerun()
        except Exception as err:
            st.error(f"خطا در به‌روزرسانی: {err}")
            cached = load_disk_cache()
            if cached is not None:
                st.session_state["market_data"] = cached
                st.session_state["is_cached_fallback"] = True
                st.warning("داده‌های قبلی از حافظه موقت لود شدند.")
            st.rerun()

    # به‌روزرسانی خودکار
    auto_refresh_on = st.toggle("فعال‌سازی رفرش خودکار", value=False)
    if auto_refresh_on:
        interval_sec = st.selectbox(
            "فاصله زمانی رفرش:",
            options=[30, 60, 120],
            format_func=lambda x: f"هر {x} ثانیه",
            index=1,
        )
        st_autorefresh(interval=interval_sec * 1000, key="market_autorefresh_timer")
        st.caption(f"⚡ رفرش خودکار هر {interval_sec} ثانیه فعال است.")

    st.markdown("---")

    # نشانگر تازگی داده (Data Freshness Indicator)
    data_pkg = st.session_state.get("market_data", {})
    df_all = data_pkg.get("df_processed", pd.DataFrame())
    u_stats_all = data_pkg.get("underlying_stats", {})
    ts_jalali = data_pkg.get("timestamp_jalali", "نامشخص")
    ts_unix = data_pkg.get("timestamp_unix", time.time())

    minutes_ago = max(0, int((time.time() - ts_unix) / 60))

    if minutes_ago < 15:
        badge_html = f'<div class="freshness-green">🟢 برخط ({minutes_ago} دقیقه پیش)</div>'
    elif minutes_ago < 60:
        badge_html = f'<div class="freshness-yellow">🟡 با تأخیر ({minutes_ago} دقیقه پیش)</div>'
    else:
        badge_html = f'<div class="freshness-red">🔴 قدیمی ({minutes_ago // 60} ساعت پیش)</div>'

    st.markdown(f"**وضعیت تازگی داده:** {badge_html}", unsafe_allow_html=True)
    st.caption(f"🕒 **آخرین ثبت:** {ts_jalali}")

    if st.session_state.get("is_cached_fallback", False):
        st.caption("⚠ داده‌ها از حافظه کش بازخوانی شده‌اند.")

    st.markdown("---")
    st.markdown("##### 📊 آمار بازار اسکن‌شده")
    total_cnt = len(df_all)
    illiquid_cnt = int((df_all["کم‌عمق"] == True).sum()) if not df_all.empty else 0
    liquid_cnt = total_cnt - illiquid_cnt

    st.metric("پوشش دارایی‌های پایه", f"{len(config.target_underlying_symbols)} نماد")
    st.metric("کل قراردادهای اسکن‌شده", f"{total_cnt:,} نماد")
    st.metric("قراردادهای نقدشونده", f"{liquid_cnt:,} نماد")
    st.metric("قراردادهای کم‌عمق / پرریسک", f"{illiquid_cnt:,} نماد")

    st.markdown("---")
    st.caption("ℹ️ مدل ارزش‌گذاری: بلک-شولز-مرتون (BSM) با نرخ بدون ریسک ۲۸٪ سالانه و نوسان‌پذیری تاریخی ۳۰ روزه.")


# ==============================================================================
# ۵. تزریق استایل CSS پویا (پالت تیره موسساتی یا روشن مدرن)
# ==============================================================================
if is_dark:
    css_theme = """
    /* پالت تیره سازمانی (Dark Navy / Charcoal) */
    .stApp {
        background-color: #0B132B !important;
        color: #E2E8F0 !important;
    }
    header, [data-testid="stHeader"] {
        background-color: #0B132B !important;
    }
    [data-testid="stSidebar"] {
        background-color: #111C38 !important;
        border-left: 1px solid #1C2B50 !important;
    }
    .entry-card {
        background: linear-gradient(135deg, #162244 0%, #1C2B54 100%);
        border: 1px solid #283C6E;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
        transition: all 0.25s ease-in-out;
    }
    .entry-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.55);
        border-color: #3B579D;
    }
    .entry-card-call {
        border-right: 6px solid #10B981 !important;
    }
    .entry-card-put {
        border-right: 6px solid #F43F5E !important;
    }
    .card-title {
        color: #60A5FA !important;
    }
    .card-text-muted {
        color: #94A3B8 !important;
    }
    .card-meta-box {
        background-color: rgba(255, 255, 255, 0.04);
        border-radius: 8px;
        padding: 8px 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .score-badge {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: #ffffff;
    }
    .freshness-green {
        display: inline-block;
        background-color: rgba(16, 185, 129, 0.18);
        color: #34D399;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 12px;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .freshness-yellow {
        display: inline-block;
        background-color: rgba(245, 158, 11, 0.18);
        color: #FBBF24;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 12px;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .freshness-red {
        display: inline-block;
        background-color: rgba(239, 68, 68, 0.18);
        color: #F87171;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 12px;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .underlying-rank-badge {
        display: inline-block;
        background: rgba(96, 165, 250, 0.16);
        color: #93C5FD;
        border: 1px solid rgba(96, 165, 250, 0.38);
        padding: 3px 11px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* نوارهای سبک مینیاتوری CSS بدون لگ Plotly */
    .subscore-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 10px;
        margin-top: 12px;
        margin-bottom: 8px;
    }
    .subscore-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 8px;
        padding: 8px 12px;
    }
    .subscore-label {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        margin-bottom: 5px;
        color: #E2E8F0;
    }
    .mini-bar-track {
        width: 100%;
        height: 6px;
        background: rgba(148, 163, 184, 0.16);
        border-radius: 4px;
        overflow: hidden;
    }
    .mini-bar-fill-call {
        height: 100%;
        background: linear-gradient(90deg, #059669, #10B981);
        border-radius: 4px;
        transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .mini-bar-fill-put {
        height: 100%;
        background: linear-gradient(90deg, #DC2626, #F43F5E);
        border-radius: 4px;
        transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* بخش قراردادهای جایگزین همنام (Sister Contracts) */
    .sister-contracts-box {
        margin-top: 10px;
        padding-top: 9px;
        border-top: 1px dashed rgba(148, 163, 184, 0.2);
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        font-size: 12.5px;
    }
    .sister-title {
        color: #94A3B8;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .sister-badges-wrapper {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
    }
    .sister-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 8px;
        padding: 4px 12px;
        color: #E2E8F0;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.2);
        transition: all 0.2s ease;
    }
    .sister-badge:hover {
        border-color: #60A5FA;
        background: rgba(30, 41, 59, 0.95);
        transform: translateY(-1px);
    }
    .sister-sym {
        font-weight: bold;
        color: #60A5FA;
        font-size: 13px;
    }
    .sister-meta {
        color: #94A3B8;
        font-size: 11.5px;
    }
    .sister-score-call {
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 6px;
        padding: 1px 7px;
        font-size: 11px;
        font-weight: bold;
    }
    .sister-score-put {
        background: rgba(244, 63, 94, 0.18);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.35);
        border-radius: 6px;
        padding: 1px 7px;
        font-size: 11px;
        font-weight: bold;
    }
    """
else:
    css_theme = """
    /* پالت روشن مدرن (Light Modern) */
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
    }
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-left: 1px solid #E2E8F0 !important;
    }
    .entry-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
        transition: all 0.25s ease-in-out;
    }
    .entry-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.10);
        border-color: #94A3B8;
    }
    .entry-card-call {
        border-right: 6px solid #059669 !important;
    }
    .entry-card-put {
        border-right: 6px solid #DC2626 !important;
    }
    .card-title {
        color: #1E40AF !important;
    }
    .card-text-muted {
        color: #64748B !important;
    }
    .card-meta-box {
        background-color: #F1F5F9;
        border-radius: 8px;
        padding: 8px 12px;
        border: 1px solid #E2E8F0;
    }
    .score-badge {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: #ffffff;
    }
    .freshness-green {
        display: inline-block;
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 12px;
        border: 1px solid #86EFAC;
    }
    .freshness-yellow {
        display: inline-block;
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 12px;
        border: 1px solid #FDE047;
    }
    .freshness-red {
        display: inline-block;
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 12px;
        border: 1px solid #FCA5A5;
    }
    .underlying-rank-badge {
        display: inline-block;
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 3px 11px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    
    /* نوارهای سبک مینیاتوری CSS بدون لگ Plotly */
    .subscore-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 10px;
        margin-top: 12px;
        margin-bottom: 8px;
    }
    .subscore-item {
        background: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 8px 12px;
    }
    .subscore-label {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        margin-bottom: 5px;
        color: #1E293B;
    }
    .mini-bar-track {
        width: 100%;
        height: 6px;
        background: #CBD5E1;
        border-radius: 4px;
        overflow: hidden;
    }
    .mini-bar-fill-call {
        height: 100%;
        background: linear-gradient(90deg, #059669, #10B981);
        border-radius: 4px;
        transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .mini-bar-fill-put {
        height: 100%;
        background: linear-gradient(90deg, #DC2626, #F43F5E);
        border-radius: 4px;
        transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* بخش قراردادهای جایگزین همنام (Sister Contracts) */
    .sister-contracts-box {
        margin-top: 10px;
        padding-top: 9px;
        border-top: 1px dashed rgba(148, 163, 184, 0.3);
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
        font-size: 12.5px;
    }
    .sister-title {
        color: #475569;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .sister-badges-wrapper {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
    }
    .sister-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        background: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 4px 12px;
        color: #1E293B;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    .sister-badge:hover {
        border-color: #3B82F6;
        background: #E2E8F0;
        transform: translateY(-1px);
    }
    .sister-sym {
        font-weight: bold;
        color: #1D4ED8;
        font-size: 13px;
    }
    .sister-meta {
        color: #64748B;
        font-size: 11.5px;
    }
    .sister-score-call {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
        border-radius: 6px;
        padding: 1px 7px;
        font-size: 11px;
        font-weight: bold;
    }
    .sister-score-put {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
        border-radius: 6px;
        padding: 1px 7px;
        font-size: 11px;
        font-weight: bold;
    }
    """

st.markdown(
    f"""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');
    
    html, body, [class*="css"], .stMarkdown, .stButton, .stDataFrame, .stSelectbox, .stSlider {{
        font-family: 'Vazirmatn', -apple-system, BlinkMacSystemFont, Tahoma, sans-serif !important;
        direction: rtl;
        text-align: right;
    }}
    
    /* سربرگ تب‌ها */
    .stTabs [data-baseweb="tab-list"] {{
        direction: rtl;
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Vazirmatn' !important;
        font-weight: bold;
        font-size: 15px;
        padding: 10px 20px;
    }}

    {css_theme}
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# ۶. تابع کمکی ترسیم چارت راداری Plotly برای کارت‌ها
# ==============================================================================
def create_spider_chart(
    readiness: float,
    relative_val: float,
    leverage: float,
    liquidity: float,
    dte_suit: float,
    is_call: bool = True,
    dark_mode: bool = True,
):
    """
    تولید چارت راداری مینیمال و جذاب Plotly جهت نمایش توازن ۵ مؤلفه امتیاز (v3)
    """
    categories = [
        "آمادگی پایه",
        "ارزش نسبی (BSM)",
        "اهرم",
        "نقدینگی ترکیبی",
        "تناسب سررسید",
    ]
    r_values = [readiness, relative_val, leverage, liquidity, dte_suit, readiness]
    cat_closed = categories + [categories[0]]

    if is_call:
        line_color = "#10B981"
        fill_color = "rgba(16, 185, 129, 0.28)" if dark_mode else "rgba(16, 185, 129, 0.22)"
    else:
        line_color = "#F43F5E"
        fill_color = "rgba(244, 63, 94, 0.28)" if dark_mode else "rgba(244, 63, 94, 0.22)"

    grid_color = "rgba(255, 255, 255, 0.12)" if dark_mode else "rgba(0, 0, 0, 0.08)"
    text_color = "#CBD5E1" if dark_mode else "#334155"

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r_values,
            theta=cat_closed,
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=line_color, width=2.5),
            marker=dict(size=5, color=line_color),
            hoverinfo="r+theta",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=8.5, color=text_color),
                gridcolor=grid_color,
                linecolor=grid_color,
            ),
            angularaxis=dict(
                tickfont=dict(size=10.5, family="Vazirmatn", color=text_color),
                gridcolor=grid_color,
                linecolor=grid_color,
                direction="clockwise",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(l=30, r=30, t=20, b=20),
        height=210,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ==============================================================================
# ۷. سربرگ و عنوان برنامه
# ==============================================================================
st.markdown(
    """
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <h2 style="margin:0; font-weight:bold;">📈 اسکنر و رتبه‌بند اختیار معامله بورس تهران</h2>
    </div>
    <div style="color:#64748B; font-size:14px; margin-bottom:18px;">
        پایش جامع ۲۴ دارایی پایه • ارزش‌گذاری بلک-شولز • رتبه‌بندی الگوریتمی نسخه v4 (گیت سخت اهرم ≥ ۳.۰) • مهار Deep OTM و سقف ضد-Chasing
    </div>
    """,
    unsafe_allow_html=True,
)

# تب‌های اصلی برنامه
tab1, tab2, tab3 = st.tabs([
    "📋 همه قراردادها (اسکن جامع)",
    "🎯 ۱۰ پیشنهاد برتر ورود",
    "📊 خلاصه ۲۴ دارایی پایه",
])


# ==============================================================================
# ۸. تب ۱: همه قراردادها (با جدول یکپارچه، فیلترها، صفحه‌بندی و مهار Deep OTM)
# ==============================================================================
with tab1:
    if df_all.empty:
        st.warning("داده‌ای برای نمایش موجود نیست.")
    else:
        # پنل فیلترهای تعاملی
        with st.expander("🔍 فیلترهای تعاملی بازار", expanded=True):
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)

            with f_col1:
                available_underlyings = sorted(df_all["دارایی پایه"].dropna().unique().tolist())
                sel_underlyings = st.multiselect(
                    "دارایی‌های پایه:",
                    options=available_underlyings,
                    default=available_underlyings,
                )

            with f_col2:
                sel_type = st.radio(
                    "نوع قرارداد:",
                    options=["همه", "اختیار خرید (Call)", "اختیار فروش (Put)"],
                    horizontal=True,
                )

            with f_col3:
                min_dte = int(df_all["روزهای تا سررسید (DTE)"].min())
                max_dte = int(df_all["روزهای تا سررسید (DTE)"].max())
                sel_dte = st.slider(
                    "بازه سررسید (DTE):",
                    min_value=min_dte,
                    max_value=max_dte,
                    value=(min_dte, max_dte),
                )

            with f_col4:
                sel_moneyness = st.multiselect(
                    "وضعیت سودآوری (Moneyness):",
                    options=["در سود (ITM)", "بی‌تفاوت (ATM)", "در زیان (OTM)"],
                    default=["در سود (ITM)", "بی‌تفاوت (ATM)", "در زیان (OTM)"],
                )

            f_col5, f_col6, f_col7 = st.columns([2, 2, 2])
            with f_col5:
                min_val_tomans = st.number_input(
                    "حداقل ارزش معامله امروز (میلیون تومان):",
                    min_value=0,
                    value=0,
                    step=50,
                )
                min_val_rials = min_val_tomans * 10_000_000.0

            with f_col6:
                search_kw = st.text_input("جستجوی نماد یا نام سهم:", "")

            with f_col7:
                hide_illiquid = st.checkbox("فقط قراردادهای نقدشونده", value=False)
                hide_deep_otm = st.checkbox("حذف عمیقاً بی‌ارزش (Deep OTM)", value=False)

        # اعمال فیلترها
        filtered_df = df_all.copy()

        if sel_underlyings:
            filtered_df = filtered_df[filtered_df["دارایی پایه"].isin(sel_underlyings)]

        if sel_type != "همه":
            filtered_df = filtered_df[filtered_df["نوع قرارداد"] == sel_type]

        filtered_df = filtered_df[
            (filtered_df["روزهای تا سررسید (DTE)"] >= sel_dte[0])
            & (filtered_df["روزهای تا سررسید (DTE)"] <= sel_dte[1])
        ]

        if sel_moneyness:
            filtered_df = filtered_df[filtered_df["وضعیت سودآوری"].isin(sel_moneyness)]

        if min_val_rials > 0:
            filtered_df = filtered_df[filtered_df["ارزش معاملات امروز (ریال)"] >= min_val_rials]

        if hide_illiquid:
            filtered_df = filtered_df[filtered_df["کم‌عمق"] == False]

        if hide_deep_otm and "عمیقاً بی‌ارزش" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["عمیقاً بی‌ارزش"] == False]

        if search_kw.strip():
            kw = search_kw.strip()
            filtered_df = filtered_df[
                filtered_df["نماد"].str.contains(kw, na=False)
                | filtered_df["نام قرارداد"].str.contains(kw, na=False)
                | filtered_df["دارایی پایه"].str.contains(kw, na=False)
            ]

        total_filtered = len(filtered_df)

        # صفحه‌بندی (Pagination)
        st.markdown("---")
        p_col1, p_col2, p_col3 = st.columns([2, 3, 2])

        with p_col1:
            page_size_opt = st.selectbox(
                "تعداد ردیف در صفحه:",
                options=[25, 50, 100, 200, "همه"],
                index=1,
            )

        if page_size_opt == "همه":
            page_size = max(total_filtered, 1)
            total_pages = 1
        else:
            page_size = int(page_size_opt)
            total_pages = max(1, int(np.ceil(total_filtered / page_size)))

        if "current_page" not in st.session_state:
            st.session_state["current_page"] = 1

        if st.session_state["current_page"] > total_pages:
            st.session_state["current_page"] = total_pages

        with p_col2:
            nav_prev, nav_info, nav_next = st.columns([1, 2, 1])
            with nav_prev:
                if st.button("⬅️ قبلی", disabled=(st.session_state["current_page"] <= 1)):
                    st.session_state["current_page"] -= 1
                    st.rerun()
            with nav_info:
                st.markdown(
                    f"<div style='text-align:center; padding-top:6px; font-weight:bold;'>"
                    f"صفحه {st.session_state['current_page']} از {total_pages} (کل: {total_filtered:,})"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with nav_next:
                if st.button("بعدی ➡️", disabled=(st.session_state["current_page"] >= total_pages)):
                    st.session_state["current_page"] += 1
                    st.rerun()

        with p_col3:
            jump_page = st.number_input(
                "برو به صفحه:",
                min_value=1,
                max_value=total_pages,
                value=st.session_state["current_page"],
                step=1,
                key="jump_page_input",
            )
            if jump_page != st.session_state["current_page"]:
                st.session_state["current_page"] = jump_page
                st.rerun()

        # برش ردیف‌های صفحه جاری
        start_idx = (st.session_state["current_page"] - 1) * page_size
        end_idx = start_idx + page_size
        paged_df = filtered_df.iloc[start_idx:end_idx].copy()

        # انتخاب ستون‌های اصلی جدول
        table_cols = [
            "نماد",
            "دارایی پایه",
            "نوع قرارداد",
            "قیمت پایانی بازار",
            "درصد تغییر امروز (%)",
            "قیمت تئوریک BSM",
            "حباب خام (%)",
            "حباب ریالی",
            "اهرم",
            "امتیاز اهرم",
            "امتیاز الگوریتم",
            "برچسب سفته‌بازی",
            "آمادگی پایه",
            "ارزش معاملات امروز (ریال)",
            "تعداد معاملات امروز",
            "روزهای تا سررسید (DTE)",
            "وضعیت سودآوری",
            "فاصله تا سربه‌سر (%)",
            "وضعیت نقدینگی",
            "روند دارایی پایه",
            "هم‌جهتی با روند",
        ]
        # ایجاد ستون‌های سازگار در صورت تغییر نام
        if "امتیاز آمادگی پایه" in paged_df.columns:
            paged_df["آمادگی پایه"] = paged_df["امتیاز آمادگی پایه"]

        active_cols = [c for c in table_cols if c in paged_df.columns]
        display_sub = paged_df[active_cols].copy()

        # نمایش جدول با هدر چسبان و پیکربندی اختصاصی
        st.dataframe(
            display_sub,
            use_container_width=True,
            height=580,
            hide_index=True,
            column_config={
                "نماد": st.column_config.TextColumn("نماد", help="نماد معاملاتی قرارداد در بورس تهران"),
                "دارایی پایه": st.column_config.TextColumn("دارایی پایه"),
                "نوع قرارداد": st.column_config.TextColumn("نوع"),
                "قیمت پایانی بازار": st.column_config.NumberColumn("قیمت پایانی", format="%,d"),
                "قیمت تئوریک BSM": st.column_config.NumberColumn("قیمت BSM", format="%,d"),
                "درصد تغییر امروز (%)": st.column_config.NumberColumn("تغییر امروز", format="%.2f%%"),
                "حباب خام (%)": st.column_config.NumberColumn("حباب (%)", format="%.1f%%", help="برای قراردادهای Deep OTM حباب درصدی به علت مخرج نزدیک صفر N/A است."),
                "حباب ریالی": st.column_config.NumberColumn("حباب ریالی", format="%,d", help="اختلاف قیمت بازار از قیمت تئوریک BSM به ریال"),
                "اهرم": st.column_config.NumberColumn("اهرم", format="%.2f"),
                "امتیاز اهرم": st.column_config.NumberColumn("امتیاز اهرم (۱۵٪)", format="%.1f", help="امتیاز صدکی اهرم (۰ تا ۱۰۰) در میان واجدین شرایط روز"),
                "امتیاز الگوریتم": st.column_config.NumberColumn("امتیاز الگوریتم (v3)", format="%.1f", help="امتیاز واحد سیستم جامع بر اساس وزن‌های v3 (۰ تا ۱۰۰). قراردادهای لاتاری/بی‌ارزش عمیق فاقد امتیاز هستند."),
                "برچسب سفته‌بازی": st.column_config.TextColumn("برچسب وضعیت / لاتاری", help="برچسب 'لاتاری/بی‌ارزش عمیق' برای قراردادهای با BSM کمتر از ۱۰ ریال یا دلتا کمتر از ۰.۱۰"),
                "آمادگی پایه": st.column_config.NumberColumn("آمادگی پایه", format="%.0f"),
                "ارزش معاملات امروز (ریال)": st.column_config.NumberColumn("ارزش معامله (ریال)", format="%,d"),
                "تعداد معاملات امروز": st.column_config.NumberColumn("تعداد معامله", format="%,d"),
                "روزهای تا سررسید (DTE)": st.column_config.NumberColumn("DTE", format="%,d"),
                "فاصله تا سربه‌سر (%)": st.column_config.NumberColumn("فاصله سربه‌سر", format="%.1f%%"),
            },
        )

        # دانلود فایل CSV
        csv_bytes = filtered_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 دانلود خروجی کامل جدول فیلترشده (CSV)",
            data=csv_bytes,
            file_name=f"options_scanner_{jdatetime.date.today().strftime('%Y-%m-%d')}.csv",
            mime="text/csv",
        )


# ==============================================================================
# ۹. تب ۲: ۱۰ پیشنهاد برتر ورود (تفکیک به ۲ زیرتب Call و Put با چارت راداری Plotly)
# ==============================================================================
with tab2:
    if df_all.empty:
        st.warning("داده‌ای برای ارزیابی موجود نیست.")
    else:
        # استخراج برترین‌های Call و Put
        df_calls_top, df_puts_top, conc_info = compute_top_call_and_put(
            df_all, config=config, top_n=10
        )

        # قدم ۵ — خودآزمایی اجباری قبل از نمایش
        val_errors = conc_info.get("validation_errors", {})
        errors_call = val_errors.get("Call", [])
        errors_put = val_errors.get("Put", [])
        if errors_call or errors_put:
            st.error("خطای اعتبارسنجی الگوریتم — نتایج فعلی قابل اعتماد نیستند:")
            for e in errors_call + errors_put:
                st.error(e)
            st.stop()  # از نمایش نتایج نادرست جلوگیری کن

        # آماده‌سازی داده‌های فایل خلاصه جامع گزینه‌های برتر (Top Choices Overview CSV)
        df_top_overview = create_top_choices_overview_df(df_calls_top, df_puts_top)
        today_jalali_str = jdatetime.date.today().strftime("%Y-%m-%d")
        overview_csv_bytes = df_top_overview.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        # نوار ابزار و گزینه‌های صدور فایل خلاصه CSV
        with st.container():
            st.markdown(
                """
                <div style="background:rgba(59, 130, 246, 0.08); border:1px solid rgba(59, 130, 246, 0.25); border-radius:10px; padding:12px 18px; margin-bottom:14px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
                        <div>
                            <span style="font-weight:bold; font-size:14.5px; color:#3B82F6;">📊 گزینه‌های خروجی فایل خلاصه گزینه‌های برتر (Overview CSV)</span>
                            <div style="font-size:12px; color:#94A3B8; margin-top:3px;">
                                امکان دریافت و ذخیره فایل اکسل/CSV تلفیقی شامل تمام پیشنهادهای برتر خرید و فروش به همراه امتیازات تفکیکی، رتبه‌ها و نمادهای جایگزین همنام.
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_csv1, col_csv2, col_csv3 = st.columns([3, 2, 2])
            with col_csv1:
                st.download_button(
                    label="📥 دانلود فایل خلاصه جامع گزینه‌های برتر (CSV)",
                    data=overview_csv_bytes,
                    file_name=f"options_top_choices_{today_jalali_str}.csv",
                    mime="text/csv",
                    key="btn_download_top_overview_csv",
                )
            with col_csv2:
                if st.button("💾 ذخیره مستقیم فایل در outputs", key="btn_save_top_overview_disk"):
                    _, saved_path, _ = export_top_choices_csv(
                        df_calls=df_calls_top,
                        df_puts=df_puts_top,
                        output_dir=config.output.directory,
                        filename_prefix=f"{config.output.filename_prefix}_top_choices",
                        jalali_date=today_jalali_str,
                    )
                    st.success(f"فایل با موفقیت ذخیره شد: `{saved_path}`")
            with col_csv3:
                show_overview_table = st.checkbox("👁️ نمایش پیش‌نمایش جدول خلاصه", value=False, key="chk_show_overview")

            if show_overview_table and not df_top_overview.empty:
                st.dataframe(df_top_overview, hide_index=True)
                st.markdown("---")

        # دو زیرتب مجزا برای Call و Put
        subtab_call, subtab_put = st.tabs([
            f"📈 ۱۰ خرید برتر (Call) [{len(df_calls_top)}]",
            f"🛡️ ۱۰ فروش برتر (Put) [{len(df_puts_top)}]",
        ])

        # تابع رندر کارت هر قرارداد
        def render_contract_cards(df_top, is_call_type: bool, conc_warning: bool, conc_sym: str, conc_count: int):
            if conc_warning and conc_sym:
                st.markdown(
                    f"""
                    <div style="margin-bottom:14px;">
                        <span style="background:rgba(239, 68, 68, 0.15); color:#EF4444; border:1px solid rgba(239, 68, 68, 0.35); padding:5px 14px; border-radius:14px; font-size:12.5px; font-weight:bold;">
                        ⚠ هشدار تمرکز: {conc_count} قرارداد برتر روی نماد «{conc_sym}» قرار دارند.
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            min_lev_cfg = conc_info.get("min_leverage_call", 3.0) if is_call_type else conc_info.get("min_leverage_put", 3.0)
            if df_top.empty:
                thresh_text = "۵۰۰ میلیون تومان و ۳۰ معامله" if is_call_type else "۱۰۰ میلیون تومان و ۱۰ معامله"
                st.info(f"هیچ قراردادی واجد شرایط فیلترهای سخت سه‌گانه (ارزش معامله بالای {thresh_text}، اهرم ≥ {min_lev_cfg:.1f}x، ۳ روز DTE و عدم حضور در Deep OTM) نشد.")
                return

            if 0 < len(df_top) < 10:
                type_name = "خرید (Call)" if is_call_type else "فروش (Put)"
                st.info(f"ℹ️ امروز فقط {len(df_top)} قرارداد واجد شرایط برای {type_name} یافت شد (به دلیل اعمال گیت‌های سخت نقدینگی، اهرم ≥ {min_lev_cfg:.1f}x و مهار Deep OTM).")

            for rank_i, (_, row_item) in enumerate(df_top.iterrows(), 1):
                sym = row_item["نماد"]
                u_sym = row_item["دارایی پایه"]
                opt_t = row_item["نوع قرارداد"]
                card_style = "entry-card-call" if is_call_type else "entry-card-put"
                fill_class = "mini-bar-fill-call" if is_call_type else "mini-bar-fill-put"

                final_sc = row_item.get("امتیاز الگوریتم")
                final_sc_display = f"{float(final_sc):.1f}" if pd.notna(final_sc) and final_sc is not None else "N/A"
                read_sc = float(row_item.get("امتیاز آمادگی پایه", 0.0))
                comb_liq_sc = float(row_item.get("امتیاز نقدینگی ترکیبی", 0.0))
                struct_liq = float(row_item.get("امتیاز نقدینگی ساختاری", 0.0))
                spike_liq = float(row_item.get("امتیاز جهش لحظه‌ای", 0.0))
                rel_val_sc = float(row_item.get("امتیاز ارزش نسبی (حباب)", 0.0))
                lev_sc = float(row_item.get("امتیاز اهرم", 0.0))
                dte_sc = float(row_item.get("امتیاز تناسب DTE", 0.0))
                why_expl = str(row_item.get("چرا این امتیاز", ""))

                price = int(row_item.get("قیمت پایانی بازار", 0))
                strike = int(row_item.get("قیمت اعمال", 0))
                dte = int(row_item.get("روزهای تا سررسید (DTE)", 0))
                lev = row_item.get("اهرم", 0.0)
                bub_raw = row_item.get("حباب خام (%)")
                bub_rial = int(row_item.get("حباب ریالی", 0))
                val_tm = round(float(row_item.get("ارزش معاملات امروز (ریال)", 0)) / 10_000_000)
                trades_n = int(row_item.get("تعداد معاملات امروز", 0))

                u_rank = row_item.get("رتبه در دارایی پایه")
                u_rank_html = f'<span class="underlying-rank-badge">رتبه {int(u_rank)}ام در بین قراردادهای {u_sym}</span>' if (pd.notna(u_rank) and u_rank) else ''

                bub_text = f"{bub_raw}%" if pd.notna(bub_raw) and bub_raw is not None else "N/A"

                # قراردادهای جایگزین همنام (Sister Contracts)
                sisters = row_item.get("قراردادهای جایگزین")
                if sisters is None or not isinstance(sisters, list):
                    sisters = []

                sister_badges_html = ""
                if sisters:
                    badges = []
                    score_class = "sister-score-call" if is_call_type else "sister-score-put"
                    for s in sisters:
                        sym_s = s.get("symbol", "")
                        strike_s = s.get("strike", 0)
                        dte_s = s.get("dte", 0)
                        score_s = s.get("score", 0.0)
                        b_html = f"""<div class="sister-badge"><span class="sister-sym">{sym_s}</span><span class="sister-meta">🎯 اعمال: <b>{strike_s:,}</b></span><span class="sister-meta">📅 سررسید: <b>{dte_s}</b> روز</span><span class="sister-score {score_class}">امتیاز: <b>{score_s:.1f}</b></span></div>"""
                        badges.append(b_html)

                    sister_badges_html = f"""
                    <div class="sister-contracts-box">
                        <span class="sister-title">🔄 قراردادهای جایگزین پیشنهادی:</span>
                        <div class="sister-badges-wrapper">
                            {''.join(badges)}
                        </div>
                    </div>
                    """

                with st.container():
                    st.markdown(
                        f"""
                        <div class="entry-card {card_style}">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:8px;">
                                <div>
                                    <span style="font-size:21px; font-weight:bold;" class="card-title">#{rank_i} {sym}</span>
                                    <span style="font-size:15px; margin-right:12px;" class="card-text-muted">({u_sym} | {opt_t})</span>
                                </div>
                                <div style="display:flex; align-items:center; gap:10px;">
                                    {u_rank_html}
                                    <div style="font-size:22px; font-weight:bold; color:{'#10B981' if is_call_type else '#F43F5E'};">
                                        امتیاز: {final_sc_display} <span style="font-size:13px; font-weight:normal;" class="card-text-muted">از ۱۰۰</span>
                                    </div>
                                </div>
                            </div>
                            <div style="display:flex; gap:16px; font-size:13.5px; margin-bottom:12px; flex-wrap:wrap;">
                                <div class="card-meta-box">💰 قیمت: <b>{price:,}</b> ریال</div>
                                <div class="card-meta-box">🎯 اعمال: <b>{strike:,}</b> ریال</div>
                                <div class="card-meta-box">📅 سررسید: <b>{dte}</b> روز</div>
                                <div class="card-meta-box">⚡ اهرم: <b>{lev}x</b></div>
                                <div class="card-meta-box">🎈 حباب BSM: <b>{bub_text}</b> ({bub_rial:+,} ر)</div>
                                <div class="card-meta-box">💳 ارزش معامله: <b>{val_tm:,}</b> م.ت ({trades_n} معامله)</div>
                            </div>
                            <div class="subscore-grid">
                                <div class="subscore-item">
                                    <div class="subscore-label">
                                        <span>آمادگی پایه (۲۵٪)</span>
                                        <b>{read_sc:.1f}</b>
                                    </div>
                                    <div class="mini-bar-track">
                                        <div class="{fill_class}" style="width: {min(max(read_sc, 0.0), 100.0)}%;"></div>
                                    </div>
                                </div>
                                <div class="subscore-item">
                                    <div class="subscore-label">
                                        <span>ارزش نسبی / حباب (۲۰٪)</span>
                                        <b>{rel_val_sc:.1f}</b>
                                    </div>
                                    <div class="mini-bar-track">
                                        <div class="{fill_class}" style="width: {min(max(rel_val_sc, 0.0), 100.0)}%;"></div>
                                    </div>
                                </div>
                                <div class="subscore-item">
                                    <div class="subscore-label">
                                        <span>اهرم (۱۵٪)</span>
                                        <b>{lev_sc:.1f}</b>
                                    </div>
                                    <div class="mini-bar-track">
                                        <div class="{fill_class}" style="width: {min(max(lev_sc, 0.0), 100.0)}%;"></div>
                                    </div>
                                </div>
                                <div class="subscore-item">
                                    <div class="subscore-label">
                                        <span title="ساختاری ۱۵٪: {struct_liq:.1f} | جهش ۱۰٪: {spike_liq:.1f}">نقدینگی ترکیبی (۲۵٪)</span>
                                        <b>{comb_liq_sc:.1f}</b>
                                    </div>
                                    <div class="mini-bar-track">
                                        <div class="{fill_class}" style="width: {min(max(comb_liq_sc, 0.0), 100.0)}%;"></div>
                                    </div>
                                </div>
                                <div class="subscore-item">
                                    <div class="subscore-label">
                                        <span>تناسب DTE (۱۵٪)</span>
                                        <b>{dte_sc:.1f}</b>
                                    </div>
                                    <div class="mini-bar-track">
                                        <div class="{fill_class}" style="width: {min(max(dte_sc, 0.0), 100.0)}%;"></div>
                                    </div>
                                </div>
                            </div>
                            <div style="margin-top:10px; font-size:12.5px; border-top:1px dashed rgba(148, 163, 184, 0.2); padding-top:8px;">
                                💡 <b>چرا این امتیاز:</b> <span class="card-text-muted">{why_expl}</span>
                            </div>
                            {sister_badges_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    with st.expander(f"📊 مشاهده نمودار راداری توازن ({sym})"):
                        fig_radar = create_spider_chart(
                            readiness=read_sc,
                            relative_val=rel_val_sc,
                            leverage=lev_sc,
                            liquidity=comb_liq_sc,
                            dte_suit=dte_sc,
                            is_call=is_call_type,
                            dark_mode=is_dark,
                        )
                        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

        with subtab_call:
            if not df_calls_top.empty:
                df_calls_csv = create_top_choices_overview_df(df_calls_top, pd.DataFrame())
                c_bytes = df_calls_csv.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="📥 دانلود فایل CSV ۱۰ خرید برتر (Call)",
                    data=c_bytes,
                    file_name=f"top_calls_{today_jalali_str}.csv",
                    mime="text/csv",
                    key="btn_download_calls_only_csv",
                )
            render_contract_cards(
                df_top=df_calls_top,
                is_call_type=True,
                conc_warning=conc_info.get("call_warning", False),
                conc_sym=conc_info.get("call_sym", ""),
                conc_count=conc_info.get("call_count", 0),
            )

        with subtab_put:
            if not df_puts_top.empty:
                df_puts_csv = create_top_choices_overview_df(pd.DataFrame(), df_puts_top)
                p_bytes = df_puts_csv.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button(
                    label="📥 دانلود فایل CSV ۱۰ فروش برتر (Put)",
                    data=p_bytes,
                    file_name=f"top_puts_{today_jalali_str}.csv",
                    mime="text/csv",
                    key="btn_download_puts_only_csv",
                )
            render_contract_cards(
                df_top=df_puts_top,
                is_call_type=False,
                conc_warning=conc_info.get("put_warning", False),
                conc_sym=conc_info.get("put_sym", ""),
                conc_count=conc_info.get("put_count", 0),
            )


# ==============================================================================
# ۱۰. تب ۳: خلاصه وضعیت و مومنتوم ۲۴ دارایی پایه
# ==============================================================================
with tab3:
    st.markdown("#### 📊 جدول تحلیلی شاخص‌های تکنیکال، صف و مومنتوم ۲۴ دارایی پایه")

    if not u_stats_all:
        st.warning("اطلاعات دارایی‌های پایه موجود نیست.")
    else:
        opt_counts = {}
        if not df_all.empty:
            for _, r in df_all.iterrows():
                u = str(r.get("دارایی پایه", ""))
                opt_counts.setdefault(u, {"calls": 0, "puts": 0, "total_val": 0.0})
                is_c = "Call" in str(r.get("نوع قرارداد", "")) or str(r.get("نماد", "")).startswith("ض")
                if is_c:
                    opt_counts[u]["calls"] += 1
                else:
                    opt_counts[u]["puts"] += 1
                opt_counts[u]["total_val"] += float(r.get("ارزش معاملات امروز (ریال)", 0.0))

        summary_rows = []
        seen = set()

        for sym, stats in u_stats_all.items():
            norm_s = stats.get("NormalizedSymbol", sym)
            if norm_s in seen:
                continue
            seen.add(norm_s)

            u_info = opt_counts.get(sym) or opt_counts.get(norm_s) or {"calls": 0, "puts": 0, "total_val": 0.0}

            summary_rows.append({
                "نماد پایه": stats.get("Symbol", sym),
                "آخرین قیمت": int(stats.get("Close", 0.0)),
                "روند و مومنتوم": stats.get("MomentumLabel", "خنثی"),
                "وضعیت صف": stats.get("QueueStatus", "متعادل"),
                "شاخص RSI (14 روزه)": stats.get("RSI14", 50.0),
                "وضعیت RSI": stats.get("RSIStatus", "عادی"),
                "الگوی پولبک": stats.get("PullbackDesc", "ندارد"),
                "بازدهی ۱ روزه (%)": stats.get("Return1D", 0.0),
                "بازدهی ۳ روزه (%)": stats.get("Return3D", 0.0),
                "بازدهی ۵ روزه (%)": stats.get("Return5D", 0.0),
                "بازدهی ۱۰ روزه (%)": stats.get("Return10D", 0.0),
                "فاصله از SMA5 (%)": stats.get("DistSMA5", 0.0),
                "فاصله از SMA20 (%)": stats.get("DistSMA20", 0.0),
                "سقف نوسان روزانه (%)": round(float(stats.get("PriceBand", 0.03)) * 100.0, 1),
                "آستانه ضد-Chasing ۳ روزه (%)": round(float(stats.get("Threshold3D", 8.35)), 2),
                "قراردادهای Call": u_info["calls"],
                "قراردادهای Put": u_info["puts"],
                "ارزش معاملات آپشن (میلیون تومان)": round(u_info["total_val"] / 10_000_000),
            })

        df_sum = pd.DataFrame(summary_rows)

        st.dataframe(
            df_sum,
            use_container_width=True,
            height=600,
            hide_index=True,
            column_config={
                "آخرین قیمت": st.column_config.NumberColumn(format="%,d"),
                "ارزش معاملات آپشن (میلیون تومان)": st.column_config.NumberColumn(format="%,d"),
                "شاخص RSI (14 روزه)": st.column_config.ProgressColumn(
                    format="%.1f",
                    min_value=0,
                    max_value=100,
                ),
                "بازدهی ۱ روزه (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "بازدهی ۳ روزه (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "بازدهی ۵ روزه (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "بازدهی ۱۰ روزه (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "فاصله از SMA5 (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "فاصله از SMA20 (%)": st.column_config.NumberColumn(format="%.2f%%"),
                "سقف نوسان روزانه (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "آستانه ضد-Chasing ۳ روزه (%)": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )


# ==============================================================================
# ۱۱. فوتر مینیمال
# ==============================================================================
st.markdown(
    """
    <div style="text-align: center; padding: 26px 0 14px 0; color: #64748B; font-size: 13px; border-top: 1px solid rgba(148, 163, 184, 0.2); margin-top: 40px;">
        صرفاً ابزار رصد — نه سیگنال معاملاتی.
    </div>
    """,
    unsafe_allow_html=True,
)
