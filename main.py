"""
اسکریپت اصلی اجرای اسکنر روزانه بازار آپشن بورس تهران (TSETMC)
امکان اجرای مستقیم دستی یا زمان‌بندی‌شده از طریق cron یا Windows Task Scheduler
"""

import sys
import os
import io
import argparse
import logging
from datetime import datetime
import jdatetime
import pandas as pd

# تنظیم خروجی کنسول به UTF-8 در محیط ویندوز
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
from analytics.unified_scoring import compute_top_call_and_put
from reports.excel_exporter import ExcelExporter
from reports.csv_exporter import export_top_choices_csv


def setup_logger(log_dir: str = "logs") -> logging.Logger:
    """تنظیمات سیستم لاگ‌گیری با قابلیت ذخیره در فایل و نمایش در کنسول"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("OptionScanner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # هندلر کنسول
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # هندلر فایل لاگ
    fh = logging.FileHandler(os.path.join(log_dir, "scanner.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def main():
    parser = argparse.ArgumentParser(
        description="اسکنر و فیلتر روزانه بازار اختیار معامله بورس تهران (TSETMC)"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="مسیر فایل پیکربندی (پیش‌فرض: config.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="مسیر پوشه ذخیره خروجی اکسل و گزارش‌ها",
    )
    parser.add_argument(
        "--symbols",
        "-s",
        nargs="+",
        default=None,
        help="فهرست نمادهای دارایی پایه هدف (جایگزین مقادیر کانفیگ)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="عدم تولید فایل خروجی CSV",
    )

    args = parser.parse_args()
    logger = setup_logger()

    logger.info("==================================================")
    logger.info("   آغاز اسکن روزانه بازار اختیار معامله بورس تهران   ")
    logger.info("==================================================")

    # ۱. بارگذاری تنظیمات
    config: AppConfig = load_config(args.config)
    if args.output_dir:
        config.output.directory = args.output_dir
    if args.symbols:
        config.target_underlying_symbols = args.symbols
    if args.no_csv:
        config.output.export_csv = False

    target_symbols = config.target_underlying_symbols
    logger.info(f"نمادهای پایه هدف ({len(target_symbols)} نماد): {', '.join(target_symbols)}")

    # ۲. ایجاد نمونه‌های دریافت داده و تحلیل
    market_fetcher = MarketFetcher(request_timeout=config.network.request_timeout)
    history_fetcher = HistoryFetcher(
        timeout=config.network.request_timeout,
        max_workers=config.network.concurrency_workers,
    )

    # ۳. دریافت اسنپ‌شات زنجیره آپشن‌ها
    df_raw = market_fetcher.fetch_options_snapshot(target_symbols)
    if df_raw.empty:
        logger.error("هیچ داده‌ای از بازار اختیار دریافت نشد. اسکریپت متوقف می‌شود.")
        return

    logger.info(f"مجموع {len(df_raw)} قرارداد اختیار معامله استخراج گردید.")

    # ۴. دریافت تاریخچه و شاخص‌های تکنیکال دارایی‌های پایه
    logger.info("در حال استخراج سابقه قیمتی، بازدهی و صف‌های دارایی‌های پایه...")
    underlying_stats = history_fetcher.fetch_all_underlying_stats(target_symbols)

    # ۵. استخراج موازی میانگین ۵ روزه ارزش معاملات برای نمادهای آپشن
    inscode_list = df_raw["InsCode"].dropna().astype(str).tolist()
    active_inscodes = df_raw[df_raw["Value"] > 0]["InsCode"].dropna().astype(str).tolist()
    options_avg_values = history_fetcher.fetch_options_5d_avg_values(
        inscode_list,
        lookback_days=config.liquidity_filter.lookback_days_avg,
        active_only_inscodes=active_inscodes,
    )

    # ۶. پردازش متریک‌ها، ارزش‌گذاری بلک-شولز، حباب، اهرم و رتبه‌بندی
    df_processed = process_options_dataframe(
        df_options=df_raw,
        underlying_stats=underlying_stats,
        options_avg_values=options_avg_values,
        config=config,
    )

    if df_processed.empty:
        logger.warning("پس از پردازش، خروجی معتبری حاصل نشد.")
        return

    # ذخیره در فایل کش جهت دسترسی فوری وب‌اپلیکیشن استریم‌لیت
    try:
        import pickle
        os.makedirs("cache", exist_ok=True)
        now_str = jdatetime.datetime.now().strftime("%Y-%m-%d ساعت %H:%M:%S")
        with open(os.path.join("cache", "market_data_cache.pkl"), "wb") as f:
            pickle.dump({
                "df_processed": df_processed,
                "underlying_stats": underlying_stats,
                "failed_symbols": [],
                "timestamp_jalali": now_str,
                "total_count": len(df_processed),
            }, f)
        logger.info("داده‌های اسکن در حافظه کش محلی وب‌اپلیکیشن نیز ذخیره شد.")
    except Exception as e:
        logger.warning(f"خطا در ذخیره کش وب‌اپ: {e}")

    # ۷. تعیین نام فایل بر اساس تاریخ شمسی روز
    jalali_today = jdatetime.date.today().strftime("%Y-%m-%d")
    out_dir = config.output.directory
    os.makedirs(out_dir, exist_ok=True)

    excel_filename = f"{config.output.filename_prefix}_{jalali_today}.xlsx"
    excel_path = os.path.join(out_dir, excel_filename)

    # ۸. صدور گزارش اکسل
    exporter = ExcelExporter()
    exporter.export(
        df_all=df_processed,
        underlying_stats=underlying_stats,
        output_filepath=excel_path,
    )

    # صدور CSV در صورت فعال بودن
    if config.output.export_csv:
        csv_filename = f"{config.output.filename_prefix}_{jalali_today}.csv"
        csv_path = os.path.join(out_dir, csv_filename)
        df_processed.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"فایل CSV شیت اصلی نیز ذخیره شد: {csv_path}")

    # صدور CSV خلاصه گزینه‌های برتر (Top Call & Put)
    if getattr(config.output, "export_top_choices_csv", True):
        df_calls_top, df_puts_top, _ = compute_top_call_and_put(df_processed, config=config, top_n=10)
        _, top_csv_path, _ = export_top_choices_csv(
            df_calls=df_calls_top,
            df_puts=df_puts_top,
            output_dir=out_dir,
            filename_prefix=f"{config.output.filename_prefix}_top_choices",
            jalali_date=jalali_today,
        )
        logger.info(f"فایل خلاصه CSV گزینه‌های برتر ذخیره شد: {top_csv_path}")

    # ۹. نمایش خلاصه‌گزارش در ترمینال
    total_count = len(df_processed)
    illiquid_count = int((df_processed["کم‌عمق"] == True).sum())
    liquid_count = total_count - illiquid_count

    print("\n" + "=" * 65)
    print(f"📊 خلاصه نتایج اسکن بازار اختیار ({jalali_today})")
    print("=" * 65)
    print(f"• کل قراردادهای اسکن‌شده: {total_count} عدد")
    print(f"• قراردادهای دارای نقدینگی مناسب: {liquid_count} عدد")
    print(f"• قراردادهای با برچسب کم‌عمق / پرریسک: {illiquid_count} عدد")
    print(f"• مسیر گزارش اکسل: {excel_path}")

    # برترین قراردادهای خرید (Call)
    calls = df_processed[df_processed["نوع قرارداد"].str.contains("Call")]
    if not calls.empty:
        print("\n🏆 ۵ اختیار خرید (Call) برتر بر اساس امتیاز ترکیبی:")
        print("-" * 65)
        top_calls = calls.head(5)
        for idx, (_, r) in enumerate(top_calls.iterrows(), 1):
            print(
                f"{idx}. {r['نماد']} ({r['دارایی پایه']}) | امتیاز: {r['امتیاز ترکیبی']} | "
                f"حباب: {r['حباب خام (%)']}% | اهرم: {r['اهرم']} | نقدینگی: {r['وضعیت نقدینگی']} | "
                f"روند پایه: {r['روند دارایی پایه']}"
            )

    # برترین قراردادهای فروش (Put)
    puts = df_processed[df_processed["نوع قرارداد"].str.contains("Put")]
    if not puts.empty:
        print("\n🛡 ۵ اختیار فروش (Put) برتر بر اساس امتیاز ترکیبی:")
        print("-" * 65)
        top_puts = puts.head(5)
        for idx, (_, r) in enumerate(top_puts.iterrows(), 1):
            print(
                f"{idx}. {r['نماد']} ({r['دارایی پایه']}) | امتیاز: {r['امتیاز ترکیبی']} | "
                f"حباب: {r['حباب خام (%)']}% | اهرم: {r['اهرم']} | نقدینگی: {r['وضعیت نقدینگی']} | "
                f"روند پایه: {r['روند دارایی پایه']}"
            )

    print("\n" + "=" * 65)
    logger.info("عملیات اسکن روزانه بازار آپشن با موفقیت تکمیل گردید.")


if __name__ == "__main__":
    main()
