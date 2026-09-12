"""
ماژول دریافت و استخراج داده‌های زنجیره بازار آپشن با پشتیبانی از algotik-tse و فالبک tse-option
"""

import logging
from typing import List, Optional
import pandas as pd

from data.normalizer import normalize_fa

logger = logging.getLogger("OptionScanner.MarketFetcher")


class MarketFetcher:
    """کلاس مسئول استخراج زنجیره معاملات اختیار معامله بورس تهران"""

    def __init__(self, request_timeout: int = 12):
        self.request_timeout = request_timeout

    def fetch_options_snapshot(self, target_underlyings: List[str]) -> pd.DataFrame:
        """
        دریافت اسنپ‌شات اتمیک تمام قراردادهای اختیار معامله و فیلتر روی دارایی‌های پایه هدف.
        در صورت بروز خطا در پکیج اصلی (algotik-tse)، فالبک به روش‌های جایگزین انجام می‌شود.
        """
        norm_targets = set(normalize_fa(s) for s in target_underlyings)
        logger.info(f"شروع دریافت اسنپ‌شات بازار برای نمادهای پایه: {list(norm_targets)}")

        # ۱. تلاش نخست: استفاده از algotik-tse
        try:
            import algotik_tse as alt
            logger.info("در حال دریافت داده‌های بازار از طریق algotik-tse...")
            df_market = alt.get_option_market(progress=False)

            if df_market is not None and not df_market.empty:
                df_market["NormUnderlying"] = df_market["UnderlyingSymbol"].apply(normalize_fa)
                filtered_df = df_market[df_market["NormUnderlying"].isin(norm_targets)].copy()

                if not filtered_df.empty:
                    logger.info(f"تعداد {len(filtered_df)} نماد آپشن مرتبط با نمادهای پایه یافت شد.")
                    # تحلیل زنجیره و محاسبه یونانی‌ها و IV با algotik-tse
                    try:
                        logger.info("در حال اجرای تحلیل زنجیره آپشن (محاسبه دلتا، IV، عمق مظنه)...")
                        analyzed = alt.analyze_option_chain(options=filtered_df, progress=False)
                        if analyzed is not None and not analyzed.empty:
                            return analyzed
                    except Exception as e:
                        logger.warning(f"خطا در تحلیل زنجیره با analyze_option_chain ({e}). از داده‌های خام استفاده می‌شود.")
                    return filtered_df
                else:
                    logger.warning("هیچ نماد اختیاری مطابق با نمادهای پایه در اسنپ‌شات algotik-tse پیدا نشد.")
        except Exception as e:
            logger.error(f"خطا در دریافت داده با algotik-tse: {e}. تلاش برای فالبک...")

        # ۲. فالبک: استفاده از tse-option در صورت امکان
        try:
            import tse_option as tso
            logger.info("تلاش برای دریافت اطلاعات با پکیج جایگزین tse-option...")
            all_chains = []
            for sym in target_underlyings:
                try:
                    chain = tso.option_chain(symbol=sym, IV=True, leverage=True, P_BSM=True)
                    if chain is not None and not chain.empty:
                        chain["UnderlyingSymbol"] = sym
                        all_chains.append(chain)
                except Exception as sym_err:
                    logger.warning(f"دریافت با tse-option برای نماد {sym} ناموفق بود: {sym_err}")

            if all_chains:
                combined = pd.concat(all_chains, ignore_index=True)
                logger.info(f"تعداد {len(combined)} ردیف از طریق tse-option با موفقیت دریافت شد.")
                return combined
        except Exception as e:
            logger.error(f"فالبک tse-option نیز ناموفق بود: {e}")

        logger.critical("دریافت داده‌های آپشن از تمام منابع با خطا مواجه شد.")
        return pd.DataFrame()
