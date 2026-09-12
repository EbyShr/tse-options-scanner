"""
ماژول تحلیل روند، بازدهی و مومنتوم دارایی پایه و تولید برچسب جهت‌دهنده
"""

from typing import Dict, Any


def determine_momentum_label(underlying_stats: Dict[str, Any]) -> str:
    """
    تعیین برچسب شفاف جهت و مومنتوم دارایی پایه بر اساس:
    - فاصله از SMA5 و SMA20
    - ترتیب میانگین‌های متحرک (SMA5 نسبت به SMA20)
    - بازدهی‌های کوتاه‌مدت (۱ روزه و ۵ روزه)
    - وضعیت صف خرید یا فروش در پایان روز معاملاتی
    
    خروجی:
    - صعودی قوی
    - صعودی ملایم
    - خنثی
    - نزولی ملایم
    - نزولی قوی
    """
    score = 0

    dist_sma5 = underlying_stats.get("DistSMA5", 0.0)
    dist_sma20 = underlying_stats.get("DistSMA20", 0.0)
    sma5 = underlying_stats.get("SMA5", 0.0)
    sma20 = underlying_stats.get("SMA20", 0.0)
    ret_1d = underlying_stats.get("Return1D", 0.0)
    ret_5d = underlying_stats.get("Return5D", 0.0)
    queue = underlying_stats.get("QueueStatus", "متعادل")

    # ۱. مقایسه قیمت با SMA5
    if dist_sma5 > 1.5:
        score += 1
    elif dist_sma5 < -1.5:
        score -= 1

    # ۲. مقایسه SMA5 با SMA20 (ترتیب صعودی یا نزولی)
    if sma5 > 0 and sma20 > 0:
        if sma5 > sma20:
            score += 1
        elif sma5 < sma20:
            score -= 1

    # ۳. ترکیب بازدهی ۱ روزه و ۵ روزه
    if ret_1d > 0.8 and ret_5d > 1.5:
        score += 1
    elif ret_1d < -0.8 and ret_5d < -1.5:
        score -= 1

    # ۴. وضعیت صف خرید/فروش در پایان جلسه
    if "صف خرید" in queue:
        score += 2
    elif "صف فروش" in queue:
        score -= 2
    elif "تقاضا" in queue:
        score += 1
    elif "عرضه" in queue:
        score -= 1

    # تفسیر نهایی بر اساس مجموع امتیاز
    if score >= 3:
        return "صعودی قوی"
    elif score in (1, 2):
        return "صعودی ملایم"
    elif score in (-1, -2):
        return "نزولی ملایم"
    elif score <= -3:
        return "نزولی قوی"
    else:
        return "خنثی"
