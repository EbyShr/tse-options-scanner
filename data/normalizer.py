"""
ماژول استانداردسازی و یکسان‌سازی کاراکترهای فارسی و عربی برای نمادهای بورس تهران
"""

import re
from typing import Optional


def normalize_fa(text: Optional[str]) -> str:
    """استانداردسازی رشته متن، تبدیل حروف عربی (ي و ك) به فارسی (ی و ک) و حذف فاصله‌های اضافه"""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # تبدیل حروف عربی به فارسی
    text = text.replace("ي", "ی")
    text = text.replace("ك", "ک")
    text = text.replace("ة", "ه")
    text = text.replace("ؤ", "و")
    text = text.replace("إ", "ا")
    text = text.replace("أ", "ا")
    text = text.replace("ء", "")
    # حذف نیم‌فاصله و فاصله‌های زاید
    text = text.replace("\u200c", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def symbols_match(sym1: Optional[str], sym2: Optional[str]) -> bool:
    """بررسی برابری دو نماد با در نظر گرفتن تفاوت‌های تایپی فارسی و عربی"""
    n1 = normalize_fa(sym1)
    n2 = normalize_fa(sym2)
    return n1 == n2 or n1 in n2 or n2 in n1
