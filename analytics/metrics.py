"""
ماژول پردازش متریک‌ها و ارجاع به پایپ‌لاین واحد الگوریتم امتیازدهی
"""

import logging
from typing import Dict, Any
import pandas as pd

from config import AppConfig
from analytics.unified_scoring import run_unified_scoring_pipeline

logger = logging.getLogger("OptionScanner.Metrics")


def process_options_dataframe(
    df_options: pd.DataFrame,
    underlying_stats: Dict[str, Dict[str, Any]],
    options_avg_values: Dict[str, float],
    config: AppConfig,
) -> pd.DataFrame:
    """
    پردازش اسنپ‌شات آپشن با خط‌لوله واحد و متمرکز الگوریتم:
    تمام محاسبات، اعمال گیت‌های سخت، امتیازدهی و رتبه‌بندی به صورت ۱۰۰٪ متمرکز در unified_scoring انجام می‌شود.
    """
    if df_options.empty:
        return pd.DataFrame()

    res = run_unified_scoring_pipeline(
        df_raw=df_options,
        underlying_stats=underlying_stats,
        options_avg_values=options_avg_values,
        config=config,
    )
    return res["df_all"]

