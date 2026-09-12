"""
ماژول صدور فایل خلاصه CSV برای برترین گزینه‌های خرید و فروش (Top Call & Put Choices)
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import jdatetime

logger = logging.getLogger("OptionScanner.CsvExporter")


def format_sister_contracts_text(sisters: Any) -> str:
    """تبدیل ساختار لیست قراردادهای جایگزین به متن خلاصه و خوانا جهت درج در ستون CSV"""
    if not isinstance(sisters, list) or not sisters:
        return "ندارد"

    parts = []
    for s in sisters:
        if isinstance(s, dict):
            sym = s.get("symbol", "")
            strike = s.get("strike", 0)
            dte = s.get("dte", 0)
            score = s.get("score", 0.0)
            parts.append(f"{sym} (اعمال: {strike:,}، DTE: {dte} روز، امتیاز: {score:.1f})")
    return " | ".join(parts) if parts else "ندارد"


def create_top_choices_overview_df(
    df_calls: pd.DataFrame,
    df_puts: pd.DataFrame,
) -> pd.DataFrame:
    """
    تولید یک دیتافریم جامع، شفاف و ساختاریافته از ۱۰ پیشنهاد برتر خرید و ۱۰ پیشنهاد برتر فروش
    شامل تمام اطلاعات کلیدی، امتیازات تفکیکی، رتبه‌ها، قراردادهای جایگزین و تحلیل امتیاز.
    """
    records = []

    def _process_pool(df_subset: pd.DataFrame, category_label: str):
        if df_subset is None or df_subset.empty:
            return
        for rank, (_, row) in enumerate(df_subset.iterrows(), 1):
            val_rials = float(row.get("ارزش معاملات امروز (ریال)", 0))
            val_tomans = round(val_rials / 10_000_000, 1)

            sisters_raw = row.get("قراردادهای جایگزین", [])
            sisters_txt = format_sister_contracts_text(sisters_raw)

            bub_raw = row.get("حباب خام (%)")
            bub_pct_val = round(float(bub_raw), 1) if pd.notna(bub_raw) and bub_raw is not None else None

            bsm_val = row.get("قیمت تئوریک BSM")
            bsm_clean = int(round(float(bsm_val))) if pd.notna(bsm_val) and bsm_val is not None else None

            delta_val = row.get("دلتا")
            delta_clean = round(float(delta_val), 3) if pd.notna(delta_val) and delta_val is not None else None

            dist_be = row.get("فاصله تا سربه‌سر (%)")
            dist_be_clean = round(float(dist_be), 2) if pd.notna(dist_be) and dist_be is not None else None

            records.append({
                "دسته رتبه‌بندی": category_label,
                "رتبه در دسته": rank,
                "نماد اختیار": str(row.get("نماد", "")),
                "دارایی پایه": str(row.get("دارایی پایه", "")),
                "نوع قرارداد": str(row.get("نوع قرارداد", "")),
                "رتبه در دارایی پایه": int(row.get("رتبه در دارایی پایه", 1)),
                "امتیاز کل سیستم (از ۱۰۰)": round(float(row.get("امتیاز الگوریتم", 0.0)), 1),
                "امتیاز آمادگی پایه (۲۵٪)": round(float(row.get("امتیاز آمادگی پایه", 0.0)), 1),
                "امتیاز ارزش نسبی حباب (۲۰٪)": round(float(row.get("امتیاز ارزش نسبی (حباب)", 0.0)), 1),
                "امتیاز اهرم (۱۵٪)": round(float(row.get("امتیاز اهرم", 0.0)), 1),
                "امتیاز نقدینگی ترکیبی (۲۵٪)": round(float(row.get("امتیاز نقدینگی ترکیبی", 0.0)), 1),
                "امتیاز نقدینگی ساختاری (۱۵٪)": round(float(row.get("امتیاز نقدینگی ساختاری", 0.0)), 1),
                "امتیاز جهش لحظه‌ای (۱۰٪)": round(float(row.get("امتیاز جهش لحظه‌ای", 0.0)), 1),
                "امتیاز تناسب سررسید (۱۵٪)": round(float(row.get("امتیاز تناسب DTE", 0.0)), 1),
                "ضریب دروازه نقدینگی": round(float(row.get("ضریب دروازه نقدینگی", 1.0)), 2),
                "قیمت پایانی بازار (ریال)": int(row.get("قیمت پایانی بازار", 0)),
                "قیمت اعمال (ریال)": int(row.get("قیمت اعمال", 0)),
                "روزهای تا سررسید (DTE)": int(row.get("روزهای تا سررسید (DTE)", 0)),
                "اهرم": round(float(row.get("اهرم", 0.0)), 2),
                "حباب خام (%)": bub_pct_val,
                "حباب ریالی": int(row.get("حباب ریالی", 0)),
                "قیمت تئوریک BSM (ریال)": bsm_clean,
                "دلتا": delta_clean,
                "ارزش معاملات امروز (ریال)": int(val_rials),
                "ارزش معاملات امروز (میلیون تومان)": val_tomans,
                "تعداد معاملات امروز": int(row.get("تعداد معاملات امروز", 0)),
                "میانگین ارزش ۵ روزه (ریال)": int(float(row.get("میانگین ارزش ۵ روزه (ریال)", 0))),
                "فاصله تا سربه‌سر (%)": dist_be_clean,
                "قراردادهای جایگزین همنام": sisters_txt,
                "چرا این امتیاز": str(row.get("چرا این امتیاز", "")),
            })

    _process_pool(df_calls, "۱۰ خرید برتر (Call)")
    _process_pool(df_puts, "۱۰ فروش برتر (Put)")

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def export_top_choices_csv(
    df_calls: pd.DataFrame,
    df_puts: pd.DataFrame,
    output_dir: str = "outputs",
    filename_prefix: str = "options_top_choices",
    jalali_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, bytes]:
    """
    تولید و ذخیره فایل CSV خلاصه گزینه‌های برتر Call و Put در دایرکتوری outputs.
    خروجی: تاپل شامل (دیتافریم، مسیر فایل ذخیره‌شده، بایت‌های با انکودینگ utf-8-sig)
    """
    if jalali_date is None:
        jalali_date = jdatetime.date.today().strftime("%Y-%m-%d")

    df_overview = create_top_choices_overview_df(df_calls, df_puts)

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{filename_prefix}_{jalali_date}.csv"
    filepath = os.path.join(output_dir, filename)

    csv_bytes = df_overview.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

    with open(filepath, "wb") as f:
        f.write(csv_bytes)

    logger.info(f"فایل CSV خلاصه گزینه‌های برتر با موفقیت ذخیره شد: {filepath}")
    return df_overview, filepath, csv_bytes
