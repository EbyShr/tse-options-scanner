"""
تست‌های واحد و اعتبارسنجی منطق تحلیلی اسکنر آپشن بورس تهران
"""

import os
import unittest
import pandas as pd
import numpy as np

from data.normalizer import normalize_fa, symbols_match
from analytics.greeks_bsm import calculate_bsm_price, calculate_leverage, evaluate_moneyness
from analytics.underlying_momentum import determine_momentum_label
from analytics.metrics import process_options_dataframe
from config import AppConfig
from reports.excel_exporter import ExcelExporter


class TestOptionScanner(unittest.TestCase):

    def test_fa_normalizer(self):
        """تست نرمال‌سازی کاراکترهای فارسی و عربی"""
        self.assertEqual(normalize_fa("فملي"), "فملی")
        self.assertEqual(normalize_fa("بانك تجارت"), "بانک تجارت")
        self.assertTrue(symbols_match("فملي", "فملی"))
        self.assertTrue(symbols_match("خودرو ", " خودرو"))

    def test_bsm_calculations(self):
        """تست فرمول ارزش‌گذاری بلک-شولز و دلتا"""
        # حالت ۱: کال در سود (ITM)
        c_price, c_delta = calculate_bsm_price(
            spot=1000.0, strike=800.0, dte=30, rate=0.25, volatility=0.30, option_type="call"
        )
        self.assertGreater(c_price, 200.0)
        self.assertGreater(c_delta, 0.5)
        self.assertLessEqual(c_delta, 1.0)

        # حالت ۲: پوت در سود (ITM)
        p_price, p_delta = calculate_bsm_price(
            spot=800.0, strike=1000.0, dte=30, rate=0.25, volatility=0.30, option_type="put"
        )
        self.assertGreater(p_price, 150.0)
        self.assertLess(p_delta, 0.0)
        self.assertGreaterEqual(p_delta, -1.0)

        # حالت ۳: سررسید (DTE = 0)
        c_exp, _ = calculate_bsm_price(spot=1000.0, strike=800.0, dte=0, option_type="call")
        self.assertEqual(c_exp, 200.0)

        p_exp, _ = calculate_bsm_price(spot=1000.0, strike=800.0, dte=0, option_type="put")
        self.assertEqual(p_exp, 0.0)

    def test_leverage_and_moneyness(self):
        """تست محاسبه اهرم و وضعیت سودآوری"""
        # اهرم با قیمت ۱۰۰ ریال، سهم ۱۰۰۰ ریال و دلتا ۰.۵
        lev = calculate_leverage(spot=1000.0, option_price=100.0, delta=0.5)
        self.assertEqual(lev, 5.0)

        # اهرم با قیمت صفر یا منفی نباید ارور دهد
        self.assertEqual(calculate_leverage(spot=1000.0, option_price=0.0, delta=0.5), 0.0)

        # تست Moneyness برای Call
        m_call, dist_c = evaluate_moneyness(spot=1100.0, strike=1000.0, option_price=150.0, option_type="call")
        self.assertEqual(m_call, "در سود (ITM)")
        # نقطه سربه‌سر = ۱۱۵۰، فاصله تا ۱۱۰۰ = (1150 - 1100) / 1100 * 100 = 4.55%
        self.assertAlmostEqual(dist_c, 4.55, places=1)

        # تست Moneyness برای Put
        m_put, dist_p = evaluate_moneyness(spot=900.0, strike=1000.0, option_price=120.0, option_type="put")
        self.assertEqual(m_put, "در سود (ITM)")

    def test_momentum_label(self):
        """تست برچسب مومنتوم و روند دارایی پایه"""
        bullish_stats = {
            "DistSMA5": 3.0,
            "DistSMA20": 4.0,
            "SMA5": 1050.0,
            "SMA20": 1000.0,
            "Return1D": 2.5,
            "Return5D": 6.0,
            "QueueStatus": "صف خرید",
        }
        label_bull = determine_momentum_label(bullish_stats)
        self.assertEqual(label_bull, "صعودی قوی")

        bearish_stats = {
            "DistSMA5": -3.0,
            "DistSMA20": -5.0,
            "SMA5": 950.0,
            "SMA20": 1000.0,
            "Return1D": -2.0,
            "Return5D": -5.0,
            "QueueStatus": "صف فروش",
        }
        label_bear = determine_momentum_label(bearish_stats)
        self.assertEqual(label_bear, "نزولی قوی")

    def test_excel_exporter_structure(self):
        """تست خروجی فایل اکسل و وجود شیت‌های سه‌گانه و راست‌چین"""
        config = AppConfig()
        test_df = pd.DataFrame([
            {
                "نماد": "ضهرم6050",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت اعمال": 68000,
                "قیمت پایه": 64544,
                "قیمت پایانی بازار": 1574,
                "قیمت آخرین معامله": 1570,
                "قیمت تئوریک BSM": 1200,
                "حباب خام (%)": 31.2,
                "جذابیت حباب (معکوس)": 75.0,
                "اهرم": 8.5,
                "رتبه صدکی اهرم": 80.0,
                "وضعیت نقدینگی": "نقدشونده / عادی",
                "رتبه صدکی نقدینگی": 90.0,
                "ارزش معاملات امروز (ریال)": 21000000000,
                "میانگین ارزش ۵ روزه (ریال)": 15000000000,
                "تعداد معاملات امروز": 230,
                "نسبت معامله به میانگین": 1.4,
                "وضعیت سودآوری": "در زیان (OTM)",
                "فاصله تا سربه‌سر (%)": 7.8,
                "روند دارایی پایه": "صعودی قوی",
                "هم‌جهتی با روند": "هم‌جهت با روند صعودی پایه ✓",
                "روزهای تا سررسید (DTE)": 7,
                "تاریخ سررسید": "1405-06-25",
                "دلتا": 0.25,
                "نوسان‌پذیری": 35.0,
                "امتیاز ترکیبی": 81.5,
                "شرح فرمول امتیاز": "(75.0×0.4) + (90.0×0.35) + (80.0×0.25)",
                "کم‌عمق": False,
            },
            {
                "نماد": "طهرم8032",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار فروش (Put)",
                "قیمت اعمال": 68000,
                "قیمت پایه": 64544,
                "قیمت پایانی بازار": 11287,
                "قیمت آخرین معامله": 11200,
                "قیمت تئوریک BSM": 10000,
                "حباب خام (%)": 12.8,
                "جذابیت حباب (معکوس)": 85.0,
                "اهرم": 4.2,
                "رتبه صدکی اهرم": 40.0,
                "وضعیت نقدینگی": "کم‌عمق / ریسک نقدشوندگی بالا",
                "رتبه صدکی نقدینگی": 15.0,
                "ارزش معاملات امروز (ریال)": 300000000,
                "میانگین ارزش ۵ روزه (ریال)": 250000000,
                "تعداد معاملات امروز": 3,
                "نسبت معامله به میانگین": 1.2,
                "وضعیت سودآوری": "در سود (ITM)",
                "فاصله تا سربه‌سر (%)": -5.2,
                "روند دارایی پایه": "صعودی قوی",
                "هم‌جهتی با روند": "خلاف روند (ریسک بالا ⚠)",
                "روزهای تا سررسید (DTE)": 70,
                "تاریخ سررسید": "1405-08-30",
                "دلتا": -0.65,
                "نوسان‌پذیری": 35.0,
                "امتیاز ترکیبی": 24.6,
                "شرح فرمول امتیاز": "(85.0×0.4) + (15.0×0.35) + (40.0×0.25) [× 0.5 جریمه کم‌عمقی]",
                "کم‌عمق": True,
            },
        ])

        u_stats = {
            "اهرم": {
                "Symbol": "اهرم",
                "NormalizedSymbol": "اهرم",
                "Close": 64544.0,
                "Return1D": 3.6,
                "Return3D": 7.7,
                "Return5D": 10.7,
                "Return10D": 15.2,
                "SMA5": 61847.0,
                "DistSMA5": 4.36,
                "SMA20": 58000.0,
                "DistSMA20": 11.28,
                "RealizedVol": 0.38,
                "QueueStatus": "صف خرید",
            }
        }

        test_out = "outputs/test_output.xlsx"
        exporter = ExcelExporter()
        saved_path = exporter.export(test_df, u_stats, test_out)

        self.assertTrue(os.path.exists(saved_path))
        import openpyxl
        wb = openpyxl.load_workbook(saved_path)
        sheet_names = wb.sheetnames
        self.assertIn("تمام قراردادها (رتبه‌بندی)", sheet_names)
        self.assertIn("قراردادهای کم‌عمق (هشدار)", sheet_names)
        self.assertIn("خلاصه دارایی‌های پایه", sheet_names)

        # بررسی راست‌چین بودن
        ws1 = wb["تمام قراردادها (رتبه‌بندی)"]
        self.assertTrue(ws1.sheet_view.rightToLeft)

    def test_dte_suitability(self):
        """تست منحنی امتیاز تناسب DTE"""
        from analytics.unified_scoring import calculate_dte_suitability_score
        # زیر ۳ روز باید صفر باشد
        self.assertEqual(calculate_dte_suitability_score(2), 0.0)
        # بازه بهینه ۱۰ تا ۴۰ روز باید ۱۰۰ باشد
        self.assertEqual(calculate_dte_suitability_score(20), 100.0)
        self.assertEqual(calculate_dte_suitability_score(10), 100.0)
        self.assertEqual(calculate_dte_suitability_score(40), 100.0)
        # روزهای خیلی دور باید کمتر از ۸۰ باشد
        self.assertLess(calculate_dte_suitability_score(80), 80.0)

    def test_deep_otm_cap_and_exclusion(self):
        """تست مهار حباب در Deep OTM، ثبت حباب ریالی و برچسب سفته‌بازی"""
        from analytics.metrics import process_options_dataframe
        from analytics.unified_scoring import calculate_score
        config = AppConfig()

        # ساخت قرارداد با BSM زیر ۱۰ ریال و دلتای ناچیز (Deep OTM)
        df_deep = pd.DataFrame([
            {
                "Symbol": "ضاهرم9999",
                "InsCode": "12345",
                "UnderlyingSymbol": "اهرم",
                "OptionType": "call",
                "Strike": 150000.0,
                "Close": 50.0,
                "Last": 50.0,
                "Yesterday": 50.0,
                "DaysToExpiry": 5,
                "Value": 6_000_000_000,
                "TradeCount": 50,
            }
        ])
        u_stats = {
            "اهرم": {
                "Close": 60000.0,
                "RealizedVol": 0.35,
                "Return1D": 0.0,
                "Return3D": 0.0,
                "Return5D": 0.0,
                "SMA5": 60000.0,
                "SMA20": 60000.0,
                "DistSMA5": 0.0,
                "DistSMA20": 0.0,
                "RSI14": 50.0,
                "QueueStatus": "متعادل",
            }
        }
        res_df = process_options_dataframe(df_deep, u_stats, {}, config)

        self.assertEqual(len(res_df), 1)
        row = res_df.iloc[0]

        # بررسی مهار حباب درصدی به None و محاسبه حباب ریالی
        self.assertTrue(row["عمیقاً بی‌ارزش"])
        self.assertIsNone(row["حباب خام (%)"])
        self.assertGreater(row["حباب ریالی"], 0)
        self.assertEqual(row["برچسب سفته‌بازی"], "لاتاری/بی‌ارزش عمیق")

        # بررسی خروج از فیلتر سخت در calculate_score و امتیاز None
        score_dict = calculate_score(row, structural_liq_pct=80.0, bubble_pct_rank=None, config=config)
        self.assertFalse(score_dict["passes_hard_filter"])
        self.assertEqual(score_dict["relative_value_score"], 0.0)
        self.assertEqual(score_dict["leverage_score"], 0.0)
        self.assertIsNone(score_dict["final_score"])
        self.assertEqual(score_dict["final_score_num"], 0.0)

    def test_anti_chasing_penalty(self):
        """تست اعمال ضریب کاهشی ۰.۷ ضد-Chasing در عبور از ۹۰٪ سقف حرکت ۳ روزه (عادی ۸.۳۵٪ و اهرمی ۱۱.۲۴٪)"""
        from analytics.unified_scoring import calculate_underlying_readiness_score
        config = AppConfig()

        # حالت ۱: نماد عادی (فملی با دامنه ۳٪): آستانه ۳‌روزه ۸.۳۵٪ است. بازدهی ۹.۰٪ باید جریمه شود
        row_chase_regular = pd.Series({
            "دارایی پایه": "فملی",
            "دامنه نوسان پایه": 0.03,
            "بازدهی ۳ روزه پایه (%)": 9.0,
            "بازدهی ۱ روزه پایه (%)": 2.5,
            "بازدهی ۵ روزه پایه (%)": 10.0,
            "فاصله پایه از SMA5 (%)": 3.0,
            "فاصله پایه از SMA20 (%)": 5.0,
            "وضعیت صف پایه": "صف خرید",
            "RSI14 پایه": 65.0,
            "وضعیت RSI پایه": "عادی",
            "پولبک پایه": False,
        })
        score_reg, reasons_reg, detail_reg = calculate_underlying_readiness_score(
            row_chase_regular, is_call=True, config=config
        )
        self.assertTrue(detail_reg["anti_chasing_applied"])
        self.assertEqual(detail_reg["threshold_3d"], 8.35)
        self.assertAlmostEqual(score_reg, round(detail_reg["raw"] * 0.70, 1), places=1)
        self.assertTrue(any("ضد-Chasing" in r for r in reasons_reg))

        # حالت ۲: نماد اهرمی (اهرم با دامنه ۴٪): آستانه ۳‌روزه ۱۱.۲۴٪ است. بازدهی ۹.۵٪ نباید جریمه شود
        row_chase_lev_safe = pd.Series({
            "دارایی پایه": "اهرم",
            "دامنه نوسان پایه": 0.04,
            "بازدهی ۳ روزه پایه (%)": 9.5,
            "بازدهی ۱ روزه پایه (%)": 2.5,
            "بازدهی ۵ روزه پایه (%)": 10.0,
            "فاصله پایه از SMA5 (%)": 3.0,
            "فاصله پایه از SMA20 (%)": 5.0,
            "وضعیت صف پایه": "صف خرید",
            "RSI14 پایه": 65.0,
            "وضعیت RSI پایه": "عادی",
            "پولبک پایه": False,
        })
        score_lev, _, detail_lev = calculate_underlying_readiness_score(
            row_chase_lev_safe, is_call=True, config=config
        )
        self.assertFalse(detail_lev["anti_chasing_applied"])
        self.assertEqual(detail_lev["threshold_3d"], 11.24)
        self.assertEqual(score_lev, round(detail_lev["raw"], 1))

        # حالت ۳: نماد اهرمی با افت ۱۲.۵٪ در ۳ روز اخیر برای Put: باید جریمه اعمال شود
        row_chase_lev_put = pd.Series({
            "دارایی پایه": "اهرم",
            "دامنه نوسان پایه": 0.04,
            "بازدهی ۳ روزه پایه (%)": -12.5,
            "بازدهی ۱ روزه پایه (%)": -3.0,
            "بازدهی ۵ روزه پایه (%)": -15.0,
            "فاصله پایه از SMA5 (%)": -4.0,
            "فاصله پایه از SMA20 (%)": -7.0,
            "وضعیت صف پایه": "صف فروش",
            "RSI14 پایه": 32.0,
            "وضعیت RSI پایه": "عادی",
            "پولبک پایه": False,
        })
        score_put, reasons_put, detail_put = calculate_underlying_readiness_score(
            row_chase_lev_put, is_call=False, config=config
        )
        self.assertTrue(detail_put["anti_chasing_applied"])
        self.assertAlmostEqual(score_put, round(detail_put["raw"] * 0.70, 1), places=1)

    def test_asymmetric_readiness_call_and_put(self):
        """تست منطق نامتقارن آمادگی دارایی پایه برای Call و Put"""
        from analytics.unified_scoring import calculate_underlying_readiness_score
        config = AppConfig()

        # سهم با آرایش کاملاً صعودی
        bullish_row = pd.Series({
            "بازدهی ۳ روزه پایه (%)": 5.0,
            "بازدهی ۱ روزه پایه (%)": 2.0,
            "بازدهی ۵ روزه پایه (%)": 8.0,
            "فاصله پایه از SMA5 (%)": 3.0,
            "فاصله پایه از SMA20 (%)": 5.0,
            "وضعیت صف پایه": "صف خرید",
            "RSI14 پایه": 44.0,
            "وضعیت RSI پایه": "خروج از اشباع فروش (چرخش مثبت)",
            "پولبک پایه": True,
        })

        # ارزیابی این سهم برای Call باید امتیاز بسیار بالا بیاورد
        call_score, _, _ = calculate_underlying_readiness_score(bullish_row, is_call=True, config=config)
        # ارزیابی همان سهم برای Put باید امتیاز پایین بیاورد
        put_score, _, _ = calculate_underlying_readiness_score(bullish_row, is_call=False, config=config)

        self.assertGreater(call_score, 80.0)
        self.assertLess(put_score, 45.0)

    def test_compute_top_call_and_put_and_hard_filters(self):
        """تست استخراج هم‌زمان ۱۰ خرید و ۱۰ فروش برتر، فیلتر سخت ۵۰۰ م.ت و ۳۰ معامله و هشدار تمرکز"""
        from analytics.unified_scoring import compute_top_call_and_put
        config = AppConfig()

        df_test = pd.DataFrame([
            {
                # واجد تمام شرایط Call (ارزش ۶۰۰ م.ت، ۵۰ معامله، DTE=20)
                "نماد": "ضاهرم1",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 2000,
                "قیمت اعمال": 60000,
                "حباب خام (%)": 5.0,
                "حباب ریالی": 100,
                "عمیقاً بی‌ارزش": False,
                "ارزش معاملات امروز (ریال)": 6_000_000_000,  # بالای ۵۰۰ م.ت
                "میانگین ارزش ۵ روزه (ریال)": 5_000_000_000,
                "روزهای تا سررسید (DTE)": 20,                # بالای ۳ روز
                "تعداد معاملات امروز": 50,                   # بالای ۳۰ معامله
                "اهرم": 5.0,                                 # بالای ۳.۰
                "نسبت معامله به میانگین": 1.2,
                "جهش فومو": False,
                "پولبک پایه": True,
                "فاصله پایه از SMA5 (%)": 2.0,
                "فاصله پایه از SMA20 (%)": 1.0,
                "وضعیت صف پایه": "صف خرید",
                "RSI14 پایه": 42.0,
                "وضعیت RSI پایه": "خروج از اشباع فروش (چرخش مثبت)",
                "بازدهی ۱ روزه پایه (%)": 2.0,
                "بازدهی ۳ روزه پایه (%)": 3.0,
                "بازدهی ۵ روزه پایه (%)": 6.0,
            },
            {
                # واجد تمام شرایط Put (ارزش ۷۰۰ م.ت، ۴۰ معامله، DTE=25، اهرم=4.5)
                "نماد": "طاهرم1",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار فروش (Put)",
                "قیمت پایانی بازار": 3000,
                "قیمت اعمال": 65000,
                "حباب خام (%)": 3.0,
                "حباب ریالی": 80,
                "عمیقاً بی‌ارزش": False,
                "ارزش معاملات امروز (ریال)": 7_000_000_000,  # بالای ۵۰۰ م.ت
                "میانگین ارزش ۵ روزه (ریال)": 6_000_000_000,
                "روزهای تا سررسید (DTE)": 25,
                "تعداد معاملات امروز": 40,
                "اهرم": 4.5,                                 # بالای ۳.۰
                "نسبت معامله به میانگین": 1.1,
                "جهش فومو": False,
                "پولبک پایه": False,
                "فاصله پایه از SMA5 (%)": -2.0,
                "فاصله پایه از SMA20 (%)": -3.0,
                "وضعیت صف پایه": "صف فروش",
                "RSI14 پایه": 68.0,
                "وضعیت RSI پایه": "عادی",
                "بازدهی ۱ روزه پایه (%)": -1.5,
                "بازدهی ۳ روزه پایه (%)": -4.0,
                "بازدهی ۵ روزه پایه (%)": -5.0,
            },
            {
                # حذف با فیلتر سخت: ارزش معامله زیر ۵۰۰ میلیون تومان (۲۰۰ م.ت)
                "نماد": "ضاهرم_حذف_ارزش",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 2000,
                "قیمت اعمال": 60000,
                "حباب خام (%)": 2.0,
                "حباب ریالی": 40,
                "عمیقاً بی‌ارزش": False,
                "ارزش معاملات امروز (ریال)": 2_000_000_000,  # کمتر از ۵ میلیارد ریال
                "میانگین ارزش ۵ روزه (ریال)": 1_000_000_000,
                "روزهای تا سررسید (DTE)": 20,
                "تعداد معاملات امروز": 50,
                "اهرم": 5.0,
                "نسبت معامله به میانگین": 1.0,
                "جهش فومو": False,
                "پولبک پایه": True,
                "فاصله پایه از SMA5 (%)": 2.0,
                "فاصله پایه از SMA20 (%)": 1.0,
                "وضعیت صف پایه": "صف خرید",
                "RSI14 پایه": 42.0,
                "وضعیت RSI پایه": "خروج از اشباع فروش (چرخش مثبت)",
                "بازدهی ۱ روزه پایه (%)": 2.0,
                "بازدهی ۳ روزه پایه (%)": 3.0,
                "بازدهی ۵ روزه پایه (%)": 5.0,
            },
            {
                # حذف با فیلتر سخت: تعداد معاملات زیر ۳۰ (۱۰ معامله)
                "نماد": "ضاهرم_حذف_تعداد",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 2000,
                "قیمت اعمال": 60000,
                "حباب خام (%)": 2.0,
                "حباب ریالی": 40,
                "عمیقاً بی‌ارزش": False,
                "ارزش معاملات امروز (ریال)": 6_000_000_000,
                "میانگین ارزش ۵ روزه (ریال)": 5_000_000_000,
                "روزهای تا سررسید (DTE)": 20,
                "تعداد معاملات امروز": 10,                   # کمتر از ۳۰ معامله
                "اهرم": 5.0,
                "نسبت معامله به میانگین": 1.0,
                "جهش فومو": False,
                "پولبک پایه": True,
                "فاصله پایه از SMA5 (%)": 2.0,
                "فاصله پایه از SMA20 (%)": 1.0,
                "وضعیت صف پایه": "صف خرید",
                "RSI14 پایه": 42.0,
                "وضعیت RSI پایه": "خروج از اشباع فروش (چرخش مثبت)",
                "بازدهی ۱ روزه پایه (%)": 2.0,
                "بازدهی ۳ روزه پایه (%)": 3.0,
                "بازدهی ۵ روزه پایه (%)": 5.0,
            },
            {
                # حذف با فیلتر سخت: DTE کمتر از ۳ روز (DTE=2)
                "نماد": "ضاهرم_حذف_DTE",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 2000,
                "قیمت اعمال": 60000,
                "حباب خام (%)": 2.0,
                "حباب ریالی": 40,
                "عمیقاً بی‌ارزش": False,
                "ارزش معاملات امروز (ریال)": 6_000_000_000,
                "میانگین ارزش ۵ روزه (ریال)": 5_000_000_000,
                "روزهای تا سررسید (DTE)": 2,                    # کمتر از ۳ روز
                "تعداد معاملات امروز": 50,
                "اهرم": 5.0,
                "نسبت معامله به میانگین": 1.0,
                "جهش فومو": False,
                "پولبک پایه": True,
                "فاصله پایه از SMA5 (%)": 2.0,
                "فاصله پایه از SMA20 (%)": 1.0,
                "وضعیت صف پایه": "صف خرید",
                "RSI14 پایه": 42.0,
                "وضعیت RSI پایه": "خروج از اشباع فروش (چرخش مثبت)",
                "بازدهی ۱ روزه پایه (%)": 2.0,
                "بازدهی ۳ روزه پایه (%)": 3.0,
                "بازدهی ۵ روزه پایه (%)": 5.0,
            },
            {
                # حذف با فیلتر سخت گیت اهرم نسخه v4: اهرم کمتر از ۳.۰ (اهرم=2.4)
                "نماد": "ضاهرم_حذف_اهرم",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 2000,
                "قیمت اعمال": 60000,
                "حباب خام (%)": 2.0,
                "حباب ریالی": 40,
                "عمیقاً بی‌ارزش": False,
                "ارزش معاملات امروز (ریال)": 6_000_000_000,
                "میانگین ارزش ۵ روزه (ریال)": 5_000_000_000,
                "روزهای تا سررسید (DTE)": 20,
                "تعداد معاملات امروز": 50,
                "اهرم": 2.4,                                 # کمتر از ۳.۰ -> حذف قطعی
                "نسبت معامله به میانگین": 1.0,
                "جهش فومو": False,
                "پولبک پایه": True,
                "فاصله پایه از SMA5 (%)": 2.0,
                "فاصله پایه از SMA20 (%)": 1.0,
                "وضعیت صف پایه": "صف خرید",
                "RSI14 پایه": 42.0,
                "وضعیت RSI پایه": "خروج از اشباع فروش (چرخش مثبت)",
                "بازدهی ۱ روزه پایه (%)": 2.0,
                "بازدهی ۳ روزه پایه (%)": 3.0,
                "بازدهی ۵ روزه پایه (%)": 5.0,
            },
        ])

        df_calls, df_puts, conc_info = compute_top_call_and_put(df_test, config, top_n=10)

        # باید دقیقاً ۱ کال و ۱ پوت انتخاب شده باشند
        self.assertEqual(len(df_calls), 1)
        self.assertEqual(df_calls.iloc[0]["نماد"], "ضاهرم1")
        self.assertEqual(len(df_puts), 1)
        self.assertEqual(df_puts.iloc[0]["نماد"], "طاهرم1")

        # بررسی وجود زیرامتیازها و توضیحات
        self.assertIn("امتیاز الگوریتم", df_calls.columns)
        self.assertIn("امتیاز آمادگی پایه", df_calls.columns)
        self.assertIn("امتیاز نقدینگی ترکیبی", df_calls.columns)
        self.assertIn("امتیاز ارزش نسبی (حباب)", df_calls.columns)
        self.assertIn("امتیاز تناسب DTE", df_calls.columns)
        self.assertIn("چرا این امتیاز", df_calls.columns)

    def test_concentration_warning_call_and_put(self):
        """تست تشخیص هشدار تمرکز بیش از ۳ نماد روی یک دارایی پایه برای Call و Put"""
        from analytics.unified_scoring import compute_top_call_and_put
        config = AppConfig()

        rows = []
        # ساخت ۴ قرارداد کال روی اهرم
        for i in range(4):
            rows.append({
                "نماد": f"ضاهرم{i}",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 2000,
                "قیمت اعمال": 60000,
                "حباب خام (%)": 5.0,
                "حباب ریالی": 100,
                "عمیقاً بی‌ارزش": False,
                "ارزش معاملات امروز (ریال)": 6_000_000_000,
                "میانگین ارزش ۵ روزه (ریال)": 5_000_000_000,
                "روزهای تا سررسید (DTE)": 20,
                "تعداد معاملات امروز": 50,
                "اهرم": 4.0,
                "نسبت معامله به میانگین": 1.0,
                "جهش فومو": False,
                "پولبک پایه": False,
                "فاصله پایه از SMA5 (%)": 1.0,
                "فاصله پایه از SMA20 (%)": 1.0,
                "وضعیت صف پایه": "متعادل",
                "RSI14 پایه": 50.0,
                "وضعیت RSI پایه": "عادی",
                "بازدهی ۱ روزه پایه (%)": 1.0,
                "بازدهی ۳ روزه پایه (%)": 2.0,
                "بازدهی ۵ روزه پایه (%)": 3.0,
            })

        df_conc = pd.DataFrame(rows)
        # برای آزمودن هشدار تمرکز (> 3 نماد پایه)، سقف تنوع را بالاتر قرار می‌دهیم تا هر ۴ نماد انتخاب شوند
        config.unified_scoring.max_per_underlying = 5
        df_calls, df_puts, conc_info = compute_top_call_and_put(df_conc, config, top_n=10)

        self.assertTrue(conc_info["call_warning"])
        self.assertEqual(conc_info["call_sym"], "اهرم")
        self.assertEqual(conc_info["call_count"], 4)
        self.assertFalse(conc_info["put_warning"])

    def test_select_top_n_diversified_and_underlying_rank(self):
        """تست سقف تنوع دارایی پایه (حداکثر ۳ نماد) و محاسبه رتبه درون دارایی پایه"""
        from analytics.unified_scoring import compute_top_call_and_put, select_top_n_diversified
        config = AppConfig()
        self.assertEqual(config.unified_scoring.max_per_underlying, 3)
        self.assertEqual(config.unified_scoring.weights.underlying_readiness, 0.25)
        self.assertEqual(config.unified_scoring.weights.relative_value, 0.20)
        self.assertEqual(config.unified_scoring.weights.leverage, 0.15)
        self.assertEqual(config.unified_scoring.weights.combined_liquidity, 0.25)
        self.assertEqual(config.unified_scoring.weights.dte_suitability, 0.15)

        # ایجاد دیتاستی با ۵ قرارداد واجد شرایط اهرم و ۲ قرارداد واجد شرایط خودرو
        rows = []
        for i in range(5):
            rows.append({
                "نماد": f"ضاهرم_{i}",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 2000 + i * 10,
                "قیمت اعمال": 60000,
                "حباب خام (%)": float(i + 1),
                "حباب ریالی": 100,
                "عمیقاً بی‌ارزش": False,
                "ارزش معاملات امروز (ریال)": 10_000_000_000 - i * 1_000_000_000,
                "میانگین ارزش ۵ روزه (ریال)": 8_000_000_000,
                "روزهای تا سررسید (DTE)": 20,
                "تعداد معاملات امروز": 60,
                "اهرم": 4.0 + float(i) * 0.2,
                "نسبت معامله به میانگین": 1.2,
                "جهش فومو": False,
                "پولبک پایه": True,
                "فاصله پایه از SMA5 (%)": 2.0,
                "فاصله پایه از SMA20 (%)": 1.0,
                "وضعیت صف پایه": "صف خرید",
                "RSI14 پایه": 45.0,
                "وضعیت RSI پایه": "عادی",
                "بازدهی ۱ روزه پایه (%)": 2.0,
                "بازدهی ۳ روزه پایه (%)": 4.0,
                "بازدهی ۵ روزه پایه (%)": 5.0,
            })
        for j in range(2):
            rows.append({
                "نماد": f"ضخود_{j}",
                "دارایی پایه": "خودرو",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 1500,
                "قیمت اعمال": 3000,
                "حباب خام (%)": 3.0,
                "حباب ریالی": 50,
                "عمیقاً بی‌ارزش": False,
                "ارزش معاملات امروز (ریال)": 6_000_000_000,
                "میانگین ارزش ۵ روزه (ریال)": 5_000_000_000,
                "روزهای تا سررسید (DTE)": 25,
                "تعداد معاملات امروز": 40,
                "اهرم": 3.8 + float(j) * 0.2,
                "نسبت معامله به میانگین": 1.0,
                "جهش فومو": False,
                "پولبک پایه": False,
                "فاصله پایه از SMA5 (%)": 1.0,
                "فاصله پایه از SMA20 (%)": 1.0,
                "وضعیت صف پایه": "متعادل",
                "RSI14 پایه": 52.0,
                "وضعیت RSI پایه": "عادی",
                "بازدهی ۱ روزه پایه (%)": 1.0,
                "بازدهی ۳ روزه پایه (%)": 2.0,
                "بازدهی ۵ روزه پایه (%)": 3.0,
            })

        df_test = pd.DataFrame(rows)
        df_calls, _, _ = compute_top_call_and_put(df_test, config, top_n=10)

        # بررسی سقف تنوع: از ۵ قرارداد اهرم، حداکثر ۳ تا باید انتخاب شده باشد
        ahram_in_top = df_calls[df_calls["دارایی پایه"] == "اهرم"]
        khodro_in_top = df_calls[df_calls["دارایی پایه"] == "خودرو"]
        self.assertEqual(len(ahram_in_top), 3)
        self.assertEqual(len(khodro_in_top), 2)
        self.assertEqual(len(df_calls), 5)

        # بررسی ستون «رتبه در دارایی پایه»
        self.assertIn("رتبه در دارایی پایه", df_calls.columns)
        ahram_ranks = list(ahram_in_top["رتبه در دارایی پایه"])
        self.assertEqual(ahram_ranks, [1, 2, 3])
        khodro_ranks = list(khodro_in_top["رتبه در دارایی پایه"])
        self.assertEqual(khodro_ranks, [1, 2])

        # بررسی کارکرد تابع select_top_n_diversified با ساختار لیست آبجکت
        class DummyContract:
            def __init__(self, sym, u, sc):
                self.symbol = sym
                self.underlying = u
                self.score = sc

        dummy_list = [
            DummyContract("ضاهرم1", "اهرم", 95),
            DummyContract("ضاهرم2", "اهرم", 90),
            DummyContract("ضاهرم3", "اهرم", 85),
            DummyContract("ضاهرم4", "اهرم", 80),
            DummyContract("ضخود1", "خودرو", 75),
        ]
        selected_dummy = select_top_n_diversified(dummy_list, n=4, max_per_underlying=2)
        self.assertEqual(len(selected_dummy), 3)  # 2 ahram + 1 khodro
        self.assertEqual([c.symbol for c in selected_dummy], ["ضاهرم1", "ضاهرم2", "ضخود1"])

    def test_put_vs_call_separated_hard_filter(self):
        """تست تفکیک آستانه فیلتر سخت Call (۵۰۰ م.ت و ۳۰ معامله) و Put (۱۰۰ م.ت و ۱۰ معامله)"""
        from analytics.unified_scoring import calculate_score
        config = AppConfig()

        # قرارداد Put با ارزش ۱۵۰ میلیون تومان (۱.۵ میلیارد ریال) و ۱۵ معامله:
        # برای Put باید مجاز باشد (حدنصاب: ۱۰۰ م.ت و ۱۰ معامله)
        put_row = pd.Series({
            "نماد": "طاهرم999",
            "نوع قرارداد": "اختیار فروش (Put)",
            "دارایی پایه": "اهرم",
            "قیمت تئوریک BSM": 1500,
            "دلتا": -0.45,
            "عمیقاً بی‌ارزش": False,
            "ارزش معاملات امروز (ریال)": 1_500_000_000,
            "میانگین ارزش ۵ روزه (ریال)": 1_200_000_000,
            "تعداد معاملات امروز": 15,
            "روزهای تا سررسید (DTE)": 15,
            "اهرم": 4.0,
            "بازدهی ۱ روزه پایه (%)": -1.0,
            "بازدهی ۳ روزه پایه (%)": -2.0,
            "بازدهی ۵ روزه پایه (%)": -3.0,
            "فاصله پایه از SMA5 (%)": -1.0,
            "فاصله پایه از SMA20 (%)": -2.0,
            "وضعیت صف پایه": "متعادل",
            "RSI14 پایه": 50.0,
            "وضعیت RSI پایه": "عادی",
            "پولبک پایه": False,
        })
        put_score_res = calculate_score(put_row, structural_liq_pct=50.0, bubble_pct_rank=50.0, config=config)
        self.assertTrue(put_score_res["passes_hard_filter"])

        # قرارداد Call با همان مشخصات (۱۵۰ م.ت و ۱۵ معامله):
        # برای Call باید رد شود (حدنصاب: ۵۰۰ م.ت و ۳۰ معامله)
        call_row = put_row.copy()
        call_row["نماد"] = "ضاهرم999"
        call_row["نوع قرارداد"] = "اختیار خرید (Call)"
        call_row["دلتا"] = 0.45

        call_score_res = calculate_score(call_row, structural_liq_pct=50.0, bubble_pct_rank=50.0, config=config)
        self.assertFalse(call_score_res["passes_hard_filter"])

        # قرارداد Put با ارزش ۸۰ میلیون تومان (زیر ۱۰۰ م.ت): باید رد شود
        put_low_val = put_row.copy()
        put_low_val["ارزش معاملات امروز (ریال)"] = 800_000_000
        res_low = calculate_score(put_low_val, structural_liq_pct=50.0, bubble_pct_rank=50.0, config=config)
        self.assertFalse(res_low["passes_hard_filter"])

        # قرارداد Put با اهرم ۲.۵ (زیر ۳.۰): باید به خاطر گیت اهرم رد شود
        put_low_lev = put_row.copy()
        put_low_lev["اهرم"] = 2.5
        res_low_lev = calculate_score(put_low_lev, structural_liq_pct=50.0, bubble_pct_rank=50.0, config=config)
        self.assertFalse(res_low_lev["passes_hard_filter"])

    def test_anti_chasing_1d_fallback(self):
        """تست فالبک ضد-Chasing به بازدهی ۱‌روزه متناسب با سقف نوسان روزانه (عادی ۲.۷٪ و اهرمی ۳.۶٪)"""
        from analytics.unified_scoring import calculate_underlying_readiness_score
        config = AppConfig()

        # حالت ۱: نماد عادی (فملی با دامنه ۳٪): آستانه ۱‌روزه ۲.۷٪ است. بازدهی ۲.۸۵٪ باید جریمه شود
        row_fallback_regular = pd.Series({
            "دارایی پایه": "فملی",
            "دامنه نوسان پایه": 0.03,
            "بازدهی ۳ روزه پایه (%)": 0.0,
            "بازدهی ۱ روزه پایه (%)": 2.85,
            "بازدهی ۵ روزه پایه (%)": 2.85,
            "فاصله پایه از SMA5 (%)": 1.5,
            "فاصله پایه از SMA20 (%)": 2.0,
            "وضعیت صف پایه": "صف خرید",
            "RSI14 پایه": 65.0,
            "وضعیت RSI پایه": "عادی",
            "پولبک پایه": False,
        })
        score_reg, reasons_reg, detail_reg = calculate_underlying_readiness_score(
            row_fallback_regular, is_call=True, config=config
        )
        self.assertTrue(detail_reg["anti_chasing_applied"])
        self.assertEqual(detail_reg["threshold_1d"], 2.70)
        self.assertAlmostEqual(score_reg, round(detail_reg["raw"] * 0.70, 1), places=1)
        self.assertTrue(any("فالبک بازده ۱ روزه" in r for r in reasons_reg))

        # حالت ۲: نماد عادی با بازده ۱ روزه ۲.۲٪: نباید جریمه شود (۲.۲٪ < ۲.۷٪)
        row_reg_safe = pd.Series({
            "دارایی پایه": "فملی",
            "دامنه نوسان پایه": 0.03,
            "بازدهی ۳ روزه پایه (%)": 0.0,
            "بازدهی ۱ روزه پایه (%)": 2.2,
            "بازدهی ۵ روزه پایه (%)": 2.2,
            "فاصله پایه از SMA5 (%)": 1.0,
            "فاصله پایه از SMA20 (%)": 1.5,
            "وضعیت صف پایه": "متعادل",
            "RSI14 پایه": 55.0,
            "وضعیت RSI پایه": "عادی",
            "پولبک پایه": False,
        })
        score_reg_safe, _, detail_safe = calculate_underlying_readiness_score(
            row_reg_safe, is_call=True, config=config
        )
        self.assertFalse(detail_safe["anti_chasing_applied"])
        self.assertEqual(score_reg_safe, round(detail_safe["raw"], 1))

        # حالت ۳: نماد اهرمی (اهرم با دامنه ۴٪): آستانه ۱‌روزه ۳.۶٪ است. بازده ۳.۰٪ نباید جریمه شود (۳.۰٪ < ۳.۶٪)
        row_lev_safe = pd.Series({
            "دارایی پایه": "اهرم",
            "دامنه نوسان پایه": 0.04,
            "بازدهی ۳ روزه پایه (%)": 0.0,
            "بازدهی ۱ روزه پایه (%)": 3.0,
            "بازدهی ۵ روزه پایه (%)": 3.0,
            "فاصله پایه از SMA5 (%)": 2.0,
            "فاصله پایه از SMA20 (%)": 3.0,
            "وضعیت صف پایه": "صف خرید",
            "RSI14 پایه": 60.0,
            "وضعیت RSI پایه": "عادی",
            "پولبک پایه": False,
        })
        score_lev_safe, _, detail_lev_safe = calculate_underlying_readiness_score(
            row_lev_safe, is_call=True, config=config
        )
        self.assertFalse(detail_lev_safe["anti_chasing_applied"])
        self.assertEqual(detail_lev_safe["threshold_1d"], 3.60)

        # حالت ۴: نماد اهرمی با بازده ۳.۸٪: باید جریمه شود (۳.۸٪ > ۳.۶٪)
        row_lev_chase = pd.Series({
            "دارایی پایه": "اهرم",
            "دامنه نوسان پایه": 0.04,
            "بازدهی ۳ روزه پایه (%)": 0.0,
            "بازدهی ۱ روزه پایه (%)": 3.8,
            "بازدهی ۵ روزه پایه (%)": 3.8,
            "فاصله پایه از SMA5 (%)": 2.5,
            "فاصله پایه از SMA20 (%)": 3.5,
            "وضعیت صف پایه": "صف خرید",
            "RSI14 پایه": 68.0,
            "وضعیت RSI پایه": "عادی",
            "پولبک پایه": False,
        })
        score_lev_chase, _, detail_lev_chase = calculate_underlying_readiness_score(
            row_lev_chase, is_call=True, config=config
        )
        self.assertTrue(detail_lev_chase["anti_chasing_applied"])
        self.assertAlmostEqual(score_lev_chase, round(detail_lev_chase["raw"] * 0.70, 1), places=1)

    def test_dynamic_chasing_threshold_calculation(self):
        """تست تابع محاسبه آستانه‌های ضد-Chasing بر اساس سقف نوسان بورس تهران"""
        from analytics.unified_scoring import get_chasing_thresholds
        config = AppConfig()

        # سهم عادی با فالبک
        th_3d_reg, th_1d_reg, band_reg = get_chasing_thresholds("فملی", config=config)
        self.assertAlmostEqual(th_3d_reg, 8.35, places=2)
        self.assertAlmostEqual(th_1d_reg, 2.70, places=2)
        self.assertEqual(band_reg, 0.03)

        # صندوق اهرمی با فالبک لیست
        th_3d_lev, th_1d_lev, band_lev = get_chasing_thresholds("اهرم", config=config)
        self.assertAlmostEqual(th_3d_lev, 11.24, places=2)
        self.assertAlmostEqual(th_1d_lev, 3.60, places=2)
        self.assertEqual(band_lev, 0.04)

        # سهم با دامنه نوسان پویای TSETMC (مثلاً ۵٪)
        th_3d_dyn, th_1d_dyn, band_dyn = get_chasing_thresholds("فرابورس", price_band=0.05, config=config)
        self.assertAlmostEqual(th_3d_dyn, 14.19, places=2)
        self.assertAlmostEqual(th_1d_dyn, 4.50, places=2)
        self.assertEqual(band_dyn, 0.05)

    def test_liquidity_spike_score_formula(self):
        """تست فرمول سقف‌دار جهش نقدینگی لحظه‌ای (سقف ۳ برابری = امتیاز ۱۰۰)"""
        from analytics.unified_scoring import calculate_combined_liquidity_score
        config = AppConfig()

        # حالت ۱: جهش ۳ برابری -> امتیاز جهش ۱۰۰
        _, _, spike_3x = calculate_combined_liquidity_score(
            structural_pct=50.0, today_volume=3000, avg_5d_volume=1000, config=config
        )
        self.assertEqual(spike_3x, 100.0)

        # حالت ۲: جهش ۵ برابری -> امتیاز جهش با سقف ۱۰۰ محدود می‌شود
        _, _, spike_5x = calculate_combined_liquidity_score(
            structural_pct=50.0, today_volume=5000, avg_5d_volume=1000, config=config
        )
        self.assertEqual(spike_5x, 100.0)

        # حالت ۳: جهش ۱.۵ برابری -> امتیاز جهش ۵۰
        _, _, spike_1_5x = calculate_combined_liquidity_score(
            structural_pct=50.0, today_volume=1500, avg_5d_volume=1000, config=config
        )
        self.assertEqual(spike_1_5x, 50.0)

        # حالت ۴: حجم امروز صفر -> امتیاز جهش ۰
        _, _, spike_0 = calculate_combined_liquidity_score(
            structural_pct=50.0, today_volume=0, avg_5d_volume=1000, config=config
        )
        self.assertEqual(spike_0, 0.0)

    def test_robustness_nan_none_bsm_and_delta(self):
        """تست تاب‌آوری الگوریتم امتیازدهی در برابر مقادیر NaN/None در BSM و دلتا"""
        from analytics.unified_scoring import calculate_score
        config = AppConfig()

        row_nan = pd.Series({
            "نماد": "ضاهرم_خطادار",
            "نوع قرارداد": "اختیار خرید (Call)",
            "دارایی پایه": "اهرم",
            "قیمت تئوریک BSM": np.nan,
            "دلتا": None,
            "عمیقاً بی‌ارزش": False,
            "ارزش معاملات امروز (ریال)": 8_000_000_000,
            "میانگین ارزش ۵ روزه (ریال)": 5_000_000_000,
            "تعداد معاملات امروز": 40,
            "روزهای تا سررسید (DTE)": 15,
            "بازدهی ۱ روزه پایه (%)": 1.0,
            "بازدهی ۳ روزه پایه (%)": 2.0,
            "بازدهی ۵ روزه پایه (%)": 3.0,
            "فاصله پایه از SMA5 (%)": 1.0,
            "فاصله پایه از SMA20 (%)": 1.0,
            "وضعیت صف پایه": "متعادل",
            "RSI14 پایه": 50.0,
            "وضعیت RSI پایه": "عادی",
            "پولبک پایه": False,
        })
        # نباید کرش کند، باید زیرامتیاز ارزش نسبی صفر و خارج از فیلتر سخت شود
        res_nan = calculate_score(row_nan, structural_liq_pct=60.0, bubble_pct_rank=None, config=config)
        self.assertFalse(res_nan["passes_hard_filter"])
        self.assertIsNone(res_nan["final_score"])
        self.assertEqual(res_nan["final_score_num"], 0.0)

    def test_get_sister_contracts(self):
        """تست استخراج نمادهای جایگزین همنام (Sister Contracts) معتبر با حداکثر ۲ نماد"""
        from analytics.unified_scoring import get_sister_contracts
        config = AppConfig()

        pool_data = [
            # نماد اصلی جاری
            {
                "نماد": "ضاهرم_اصلی",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 2100,
                "قیمت اعمال": 60000,
                "روزهای تا سررسید (DTE)": 20,
                "امتیاز الگوریتم": 90.0,
                "واجد فیلتر سخت": True,
                "عمیقاً بی‌ارزش": False,
            },
            # جایگزین ۱ (امتیاز ۸۸ - باید انتخاب شود)
            {
                "نماد": "ضاهرم_جایگزین۱",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 1800,
                "قیمت اعمال": 62000,
                "روزهای تا سررسید (DTE)": 25,
                "امتیاز الگوریتم": 88.0,
                "واجد فیلتر سخت": True,
                "عمیقاً بی‌ارزش": False,
            },
            # جایگزین ۲ (امتیاز ۸۵ - باید انتخاب شود)
            {
                "نماد": "ضاهرم_جایگزین۲",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 1500,
                "قیمت اعمال": 64000,
                "روزهای تا سررسید (DTE)": 30,
                "امتیاز الگوریتم": 85.0,
                "واجد فیلتر سخت": True,
                "عمیقاً بی‌ارزش": False,
            },
            # جایگزین ۳ (امتیاز ۸۰ - چون سقف ۲ است، نباید انتخاب شود)
            {
                "نماد": "ضاهرم_جایگزین۳",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 1200,
                "قیمت اعمال": 66000,
                "روزهای تا سررسید (DTE)": 35,
                "امتیاز الگوریتم": 80.0,
                "واجد فیلتر سخت": True,
                "عمیقاً بی‌ارزش": False,
            },
            # نماد اهرم با امتیاز بالا ولی فاقد فیلتر سخت (نباید انتخاب شود)
            {
                "نماد": "ضاهرم_رد_شده",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 2500,
                "قیمت اعمال": 58000,
                "روزهای تا سررسید (DTE)": 1,
                "امتیاز الگوریتم": 95.0,
                "واجد فیلتر سخت": False,
                "عمیقاً بی‌ارزش": False,
            },
            # نماد اهرم از نوع Put (چون ما دنبال Call هستیم، نباید بیاید)
            {
                "نماد": "طاهرم_فروش",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار فروش (Put)",
                "قیمت پایانی بازار": 1000,
                "قیمت اعمال": 50000,
                "روزهای تا سررسید (DTE)": 20,
                "امتیاز الگوریتم": 89.0,
                "واجد فیلتر سخت": True,
                "عمیقاً بی‌ارزش": False,
            },
            # نماد روی سهم دیگر (خودرو)
            {
                "نماد": "ضخود۱",
                "دارایی پایه": "خودرو",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 1000,
                "قیمت اعمال": 3000,
                "روزهای تا سررسید (DTE)": 20,
                "امتیاز الگوریتم": 92.0,
                "واجد فیلتر سخت": True,
                "عمیقاً بی‌ارزش": False,
            },
        ]
        df_pool = pd.DataFrame(pool_data)

        # تست ۱: استخراج جایگزین برای ضاهرم_اصلی
        sisters = get_sister_contracts(
            current_symbol="ضاهرم_اصلی",
            underlying="اهرم",
            df_pool=df_pool,
            is_call=True,
            max_sisters=2,
        )
        self.assertEqual(len(sisters), 2)
        self.assertEqual(sisters[0]["symbol"], "ضاهرم_جایگزین۱")
        self.assertEqual(sisters[0]["score"], 88.0)
        self.assertEqual(sisters[0]["strike"], 62000)
        self.assertEqual(sisters[0]["dte"], 25)
        self.assertEqual(sisters[1]["symbol"], "ضاهرم_جایگزین۲")
        self.assertEqual(sisters[1]["score"], 85.0)

        # تست ۲: نمادی که هیچ همتای معتبری ندارد (خودرو تنها ۱ نماد دارد)
        sisters_khodro = get_sister_contracts(
            current_symbol="ضخود۱",
            underlying="خودرو",
            df_pool=df_pool,
            is_call=True,
            max_sisters=2,
        )
        self.assertEqual(sisters_khodro, [])

    def test_top_choices_csv_exporter(self):
        """تست تولید و صدور فایل خلاصه جامع گزینه‌های برتر (Top Call & Put CSV)"""
        import tempfile
        from reports.csv_exporter import create_top_choices_overview_df, export_top_choices_csv, format_sister_contracts_text

        # تست فرمت خواهرها
        self.assertEqual(format_sister_contracts_text([]), "ندارد")
        self.assertEqual(
            format_sister_contracts_text([{"symbol": "ضاهرم2", "strike": 60000, "dte": 20, "score": 85.2}]),
            "ضاهرم2 (اعمال: 60,000، DTE: 20 روز، امتیاز: 85.2)",
        )

        # ساخت داده‌های تستی Call و Put
        df_call = pd.DataFrame([{
            "نماد": "ضخود7131",
            "دارایی پایه": "خودرو",
            "نوع قرارداد": "اختیار خرید (Call)",
            "رتبه در دارایی پایه": 1,
            "امتیاز الگوریتم": 75.4,
            "امتیاز آمادگی پایه": 80.0,
            "امتیاز نقدینگی ترکیبی": 78.5,
            "امتیاز نقدینگی ساختاری": 75.0,
            "امتیاز جهش لحظه‌ای": 85.0,
            "امتیاز ارزش نسبی (حباب)": 72.0,
            "امتیاز تناسب DTE": 95.0,
            "قیمت پایانی بازار": 1500,
            "قیمت اعمال": 3000,
            "روزهای تا سررسید (DTE)": 20,
            "اهرم": 3.4,
            "حباب خام (%)": -1.2,
            "حباب ریالی": -50,
            "قیمت تئوریک BSM": 1550,
            "دلتا": 0.45,
            "ارزش معاملات امروز (ریال)": 6_000_000_000,
            "تعداد معاملات امروز": 50,
            "میانگین ارزش ۵ روزه (ریال)": 5_000_000_000,
            "فاصله تا سربه‌سر (%)": 2.1,
            "قراردادهای جایگزین": [{"symbol": "ضخود7132", "strike": 550, "dte": 20, "score": 72.3}],
            "چرا این امتیاز": "نقدینگی مطلوب؛ قیمت‌گذاری منصفانه.",
        }])

        df_put = pd.DataFrame([{
            "نماد": "طخود7138",
            "دارایی پایه": "خودرو",
            "نوع قرارداد": "اختیار فروش (Put)",
            "رتبه در دارایی پایه": 1,
            "امتیاز الگوریتم": 66.3,
            "امتیاز آمادگی پایه": 65.0,
            "امتیاز نقدینگی ترکیبی": 70.0,
            "امتیاز نقدینگی ساختاری": 68.0,
            "امتیاز جهش لحظه‌ای": 75.0,
            "امتیاز ارزش نسبی (حباب)": 60.0,
            "امتیاز تناسب DTE": 90.0,
            "قیمت پایانی بازار": 1200,
            "قیمت اعمال": 750,
            "روزهای تا سررسید (DTE)": 20,
            "اهرم": 2.8,
            "حباب خام (%)": 0.5,
            "حباب ریالی": 20,
            "قیمت تئوریک BSM": 1180,
            "دلتا": -0.35,
            "ارزش معاملات امروز (ریال)": 2_000_000_000,
            "تعداد معاملات امروز": 20,
            "میانگین ارزش ۵ روزه (ریال)": 1_800_000_000,
            "فاصله تا سربه‌سر (%)": -1.5,
            "قراردادهای جایگزین": [],
            "چرا این امتیاز": "حباب معقول؛ تناسب سررسید.",
        }])

        # ۱. بررسی تولید DataFrame خلاصه
        df_overview = create_top_choices_overview_df(df_call, df_put)
        self.assertEqual(len(df_overview), 2)
        self.assertIn("دسته رتبه‌بندی", df_overview.columns)
        self.assertIn("رتبه در دسته", df_overview.columns)
        self.assertIn("قراردادهای جایگزین همنام", df_overview.columns)
        self.assertEqual(df_overview.iloc[0]["دسته رتبه‌بندی"], "۱۰ خرید برتر (Call)")
        self.assertEqual(df_overview.iloc[1]["دسته رتبه‌بندی"], "۱۰ فروش برتر (Put)")
        self.assertIn("ضخود7132", df_overview.iloc[0]["قراردادهای جایگزین همنام"])
        self.assertEqual(df_overview.iloc[1]["قراردادهای جایگزین همنام"], "ندارد")

        # ۲. بررسی ذخیره فایل CSV روی دیسک با utf-8-sig
        with tempfile.TemporaryDirectory() as tmpdir:
            df_res, path, bts = export_top_choices_csv(
                df_calls=df_call,
                df_puts=df_put,
                output_dir=tmpdir,
                filename_prefix="test_top",
                jalali_date="1405-06-20",
            )
            self.assertTrue(os.path.exists(path))
            self.assertTrue(path.endswith("test_top_1405-06-20.csv"))
            # بررسی هدر BOM انکودینگ utf-8-sig برای پشتیبانی کامل اکسل در ویندوز
            self.assertTrue(bts.startswith(b"\xef\xbb\xbf"))
            # بررسی محتوای فایل ذخیره‌شده
            with open(path, "r", encoding="utf-8-sig") as f:
                content = f.read()
                self.assertIn("ضخود7131", content)
                self.assertIn("طخود7138", content)
                self.assertIn("۱۰ خرید برتر (Call)", content)

    def test_config_output_top_csv_option(self):
        """تست کانفیگ برای گزینه ذخیره فایل خلاصه CSV گزینه‌های برتر"""
        from config import load_config
        config = load_config()
        self.assertTrue(hasattr(config.output, "export_top_choices_csv"))
        self.assertTrue(config.output.export_top_choices_csv)

    def test_v3_scoring_algorithm_features(self):
        """تست جامع ویژگی‌های جدید نسخه v3: دروازه نقدینگی ضربی، رتبه‌بندی اهرم صدکی و حذف Deep OTM"""
        from analytics.unified_scoring import calculate_score, compute_top_call_and_put
        from reports.csv_exporter import create_top_choices_overview_df
        config = AppConfig()

        # ۱. تست دروازه نقدینگی ضربی:
        # حالت الف: معاملات امروز = ۰ (باید جریمه ۰.۱۵x اعمال شود حتی اگر نقدینگی ساختاری داشته باشد)
        row_zero_trades = pd.Series({
            "نماد": "ضاهرم_صفر_معامله",
            "دارایی پایه": "اهرم",
            "نوع قرارداد": "اختیار خرید (Call)",
            "قیمت تئوریک BSM": 2000,
            "دلتا": 0.50,
            "عمیقاً بی‌ارزش": False,
            "ارزش معاملات امروز (ریال)": 0,
            "میانگین ارزش ۵ روزه (ریال)": 5_000_000_000,
            "تعداد معاملات امروز": 0,
            "روزهای تا سررسید (DTE)": 20,
            "بازدهی ۱ روزه پایه (%)": 2.0,
            "بازدهی ۳ روزه پایه (%)": 4.0,
            "بازدهی ۵ روزه پایه (%)": 6.0,
            "فاصله پایه از SMA5 (%)": 2.0,
            "فاصله پایه از SMA20 (%)": 1.0,
            "وضعیت صف پایه": "صف خرید",
            "RSI14 پایه": 50.0,
            "وضعیت RSI پایه": "عادی",
            "پولبک پایه": False,
        })
        res_zero = calculate_score(row_zero_trades, structural_liq_pct=50.0, bubble_pct_rank=60.0, config=config, leverage_pct_rank=70.0)
        self.assertEqual(res_zero["liquidity_gate_multiplier"], 0.15)
        self.assertAlmostEqual(res_zero["final_score"], round(res_zero["raw_final_score"] * 0.15, 1), places=1)
        self.assertIn("فاقد معامله امروز", res_zero["explanation"])

        # حالت ب: نقدینگی ترکیبی < 15
        row_low_liq = row_zero_trades.copy()
        row_low_liq["تعداد معاملات امروز"] = 1
        row_low_liq["ارزش معاملات امروز (ریال)"] = 10_000_000
        res_low_liq = calculate_score(row_low_liq, structural_liq_pct=10.0, bubble_pct_rank=60.0, config=config, leverage_pct_rank=70.0)
        self.assertEqual(res_low_liq["liquidity_gate_multiplier"], 0.15)
        self.assertAlmostEqual(res_low_liq["final_score"], round(res_low_liq["raw_final_score"] * 0.15, 1), places=1)

        # حالت ج: نقدینگی نرم بین ۱۵ و ۴۰ (مثلاً نقدینگی ترکیبی = ۳۰ -> ضریب = 0.5 + 30/200 = 0.65)
        row_mid_liq = row_zero_trades.copy()
        row_mid_liq["تعداد معاملات امروز"] = 10
        row_mid_liq["ارزش معاملات امروز (ریال)"] = 200_000_000
        # structural 50 * 0.6 + spike 0 * 0.4 = 30
        res_mid_liq = calculate_score(row_mid_liq, structural_liq_pct=50.0, bubble_pct_rank=60.0, config=config, leverage_pct_rank=70.0)
        expected_mult = 0.5 + res_mid_liq["combined_liquidity_score"] / 200.0
        self.assertAlmostEqual(res_mid_liq["liquidity_gate_multiplier"], expected_mult, places=2)
        self.assertAlmostEqual(res_mid_liq["final_score"], round(res_mid_liq["raw_final_score"] * expected_mult, 1), places=1)

        # ۲. تست خروج قطعی قرارداد Deep OTM از Top 10 حتی با حجم بالا
        df_top_test = pd.DataFrame([
            {
                # قرارداد Deep OTM (BSM = 5 ریال) با معاملات و نقدینگی بسیار بالا
                "نماد": "ضاهرم_لاتاری_حجم_بالا",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 20,
                "قیمت تئوریک BSM": 5,
                "دلتا": 0.05,
                "عمیقاً بی‌ارزش": True,
                "قیمت اعمال": 120000,
                "ارزش معاملات امروز (ریال)": 50_000_000_000,
                "میانگین ارزش ۵ روزه (ریال)": 30_000_000_000,
                "تعداد معاملات امروز": 500,
                "روزهای تا سررسید (DTE)": 15,
                "اهرم": 50.0,
                "حباب خام (%)": None,
                "حباب ریالی": 15,
                "بازدهی ۱ روزه پایه (%)": 2.0,
                "بازدهی ۳ روزه پایه (%)": 4.0,
                "بازدهی ۵ روزه پایه (%)": 6.0,
                "فاصله پایه از SMA5 (%)": 2.0,
                "فاصله پایه از SMA20 (%)": 1.0,
                "وضعیت صف پایه": "صف خرید",
                "RSI14 پایه": 50.0,
                "وضعیت RSI پایه": "عادی",
                "پولبک پایه": False,
            },
            {
                # قرارداد سالم واجد شرایط
                "نماد": "ضاهرم_عادی_سالم",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 1800,
                "قیمت تئوریک BSM": 1750,
                "دلتا": 0.45,
                "عمیقاً بی‌ارزش": False,
                "قیمت اعمال": 60000,
                "ارزش معاملات امروز (ریال)": 8_000_000_000,
                "میانگین ارزش ۵ روزه (ریال)": 6_000_000_000,
                "تعداد معاملات امروز": 60,
                "روزهای تا سررسید (DTE)": 20,
                "اهرم": 5.2,
                "حباب خام (%)": 2.8,
                "حباب ریالی": 50,
                "بازدهی ۱ روزه پایه (%)": 2.0,
                "بازدهی ۳ روزه پایه (%)": 4.0,
                "بازدهی ۵ روزه پایه (%)": 6.0,
                "فاصله پایه از SMA5 (%)": 2.0,
                "فاصله پایه از SMA20 (%)": 1.0,
                "وضعیت صف پایه": "صف خرید",
                "RSI14 پایه": 50.0,
                "وضعیت RSI پایه": "عادی",
                "پولبک پایه": False,
            },
        ])
        df_c_top, _, _ = compute_top_call_and_put(df_top_test, config, top_n=10)
        # فقط قرارداد سالم باید وارد Top شود؛ قرارداد Deep OTM به هیچ وجه نباید وارد شود
        self.assertEqual(len(df_c_top), 1)
        self.assertEqual(df_c_top.iloc[0]["نماد"], "ضاهرم_عادی_سالم")

        # ۳. تست ستون‌های خروجی Overview CSV برای نسخه v3
        df_ov = create_top_choices_overview_df(df_c_top, pd.DataFrame())
        self.assertIn("امتیاز اهرم (۱۵٪)", df_ov.columns)
        self.assertIn("امتیاز ارزش نسبی حباب (۲۰٪)", df_ov.columns)
        self.assertIn("امتیاز نقدینگی ترکیبی (۲۵٪)", df_ov.columns)
        self.assertIn("ضریب دروازه نقدینگی", df_ov.columns)

    def test_v4_scoring_algorithm_features(self):
        """تست جامع بازنگری الگوریتم نسخه v4:
        ۱. گیت سخت اهرم (MIN_LEVERAGE >= 3.0 برای Call و Put)
        ۲. رتبه‌بندی صدکی اهرم منحصراً در میان بازماندگان ۳ گیت (eligible_contracts)
        ۳. نمایش تعداد واقعی قراردادهای واجد شرایط Put (عدم پر کردن مصنوعی تا ۱۰)
        ۴. ثبت لاگ هشدار در صورت کمتر بودن قراردادهای واجد شرایط Put از ۵ عدد
        """
        from analytics.unified_scoring import calculate_score, compute_top_call_and_put
        config = AppConfig()

        # ۱. تست تک‌به‌تک calculate_score برای مرز اهرم ۳.۰
        base_call = pd.Series({
            "نماد": "ضاهرم_تست_اهرم",
            "نوع قرارداد": "اختیار خرید (Call)",
            "دارایی پایه": "اهرم",
            "قیمت تئوریک BSM": 2000,
            "دلتا": 0.40,
            "عمیقاً بی‌ارزش": False,
            "ارزش معاملات امروز (ریال)": 6_000_000_000,
            "میانگین ارزش ۵ روزه (ریال)": 5_000_000_000,
            "تعداد معاملات امروز": 40,
            "روزهای تا سررسید (DTE)": 20,
            "اهرم": 2.99,  # زیر ۳.۰
            "بازدهی ۱ روزه پایه (%)": 1.0,
            "بازدهی ۳ روزه پایه (%)": 2.0,
            "بازدهی ۵ روزه پایه (%)": 3.0,
            "فاصله پایه از SMA5 (%)": 1.0,
            "فاصله پایه از SMA20 (%)": 1.0,
            "وضعیت صف پایه": "متعادل",
            "RSI14 پایه": 50.0,
            "وضعیت RSI پایه": "عادی",
            "پولبک پایه": False,
        })
        res_call_low = calculate_score(base_call, structural_liq_pct=50.0, bubble_pct_rank=50.0, config=config, leverage_pct_rank=50.0)
        self.assertFalse(res_call_low["passes_hard_filter"], "اهرم ۲.۹۹ نباید از فیلتر سخت عبور کند")

        base_call_ok = base_call.copy()
        base_call_ok["اهرم"] = 3.00
        res_call_ok = calculate_score(base_call_ok, structural_liq_pct=50.0, bubble_pct_rank=50.0, config=config, leverage_pct_rank=50.0)
        self.assertTrue(res_call_ok["passes_hard_filter"], "اهرم ۳.۰۰ باید از فیلتر سخت عبور کند")

        # ۲. دیتافریم جامع حاوی قراردادهای زیر ۳.۰ و بالای ۳.۰
        df_v4 = pd.DataFrame([
            # Call با اهرم زیر ۳ (۲.۵) و نقدینگی بالا
            {
                "نماد": "ضاهرم_رد_اهرم_پایین",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 2500,
                "قیمت تئوریک BSM": 2400,
                "دلتا": 0.55,
                "عمیقاً بی‌ارزش": False,
                "قیمت اعمال": 20000,
                "ارزش معاملات امروز (ریال)": 20_000_000_000,
                "میانگین ارزش ۵ روزه (ریال)": 15_000_000_000,
                "تعداد معاملات امروز": 200,
                "روزهای تا سررسید (DTE)": 25,
                "اهرم": 2.5,
                "حباب خام (%)": 4.1,
                "حباب ریالی": 100,
                "بازدهی ۱ روزه پایه (%)": 1.5,
                "بازدهی ۳ روزه پایه (%)": 3.0,
                "بازدهی ۵ روزه پایه (%)": 4.0,
                "فاصله پایه از SMA5 (%)": 1.0,
                "فاصله پایه از SMA20 (%)": 1.0,
                "وضعیت صف پایه": "صف خرید",
                "RSI14 پایه": 55.0,
                "وضعیت RSI پایه": "عادی",
                "پولبک پایه": False,
            },
            # Call واجد شرایط با اهرم ۳.۵
            {
                "نماد": "ضاهرم_قبول_اهرم_متوسط",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 1500,
                "قیمت تئوریک BSM": 1450,
                "دلتا": 0.45,
                "عمیقاً بی‌ارزش": False,
                "قیمت اعمال": 24000,
                "ارزش معاملات امروز (ریال)": 10_000_000_000,
                "میانگین ارزش ۵ روزه (ریال)": 8_000_000_000,
                "تعداد معاملات امروز": 90,
                "روزهای تا سررسید (DTE)": 20,
                "اهرم": 3.5,
                "حباب خام (%)": 3.4,
                "حباب ریالی": 50,
                "بازدهی ۱ روزه پایه (%)": 1.5,
                "بازدهی ۳ روزه پایه (%)": 3.0,
                "بازدهی ۵ روزه پایه (%)": 4.0,
                "فاصله پایه از SMA5 (%)": 1.0,
                "فاصله پایه از SMA20 (%)": 1.0,
                "وضعیت صف پایه": "صف خرید",
                "RSI14 پایه": 55.0,
                "وضعیت RSI پایه": "عادی",
                "پولبک پایه": False,
            },
            # Call واجد شرایط با اهرم ۶.۵ (باید رتبه صدکی اهرم بالاتری بگیرد)
            {
                "نماد": "ضاهرم_قبول_اهرم_بالا",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار خرید (Call)",
                "قیمت پایانی بازار": 800,
                "قیمت تئوریک BSM": 780,
                "دلتا": 0.35,
                "عمیقاً بی‌ارزش": False,
                "قیمت اعمال": 28000,
                "ارزش معاملات امروز (ریال)": 9_000_000_000,
                "میانگین ارزش ۵ روزه (ریال)": 7_000_000_000,
                "تعداد معاملات امروز": 80,
                "روزهای تا سررسید (DTE)": 18,
                "اهرم": 6.5,
                "حباب خام (%)": 2.5,
                "حباب ریالی": 20,
                "بازدهی ۱ روزه پایه (%)": 1.5,
                "بازدهی ۳ روزه پایه (%)": 3.0,
                "بازدهی ۵ روزه پایه (%)": 4.0,
                "فاصله پایه از SMA5 (%)": 1.0,
                "فاصله پایه از SMA20 (%)": 1.0,
                "وضعیت صف پایه": "صف خرید",
                "RSI14 پایه": 55.0,
                "وضعیت RSI پایه": "عادی",
                "پولبک پایه": False,
            },
            # Put واجد شرایط ۱ (اهرم ۴.۰)
            {
                "نماد": "طاهرم_قبول_۱",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار فروش (Put)",
                "قیمت پایانی بازار": 1200,
                "قیمت تئوریک BSM": 1180,
                "دلتا": -0.40,
                "عمیقاً بی‌ارزش": False,
                "قیمت اعمال": 22000,
                "ارزش معاملات امروز (ریال)": 2_000_000_000,
                "میانگین ارزش ۵ روزه (ریال)": 1_800_000_000,
                "تعداد معاملات امروز": 20,
                "روزهای تا سررسید (DTE)": 16,
                "اهرم": 4.0,
                "حباب خام (%)": 1.7,
                "حباب ریالی": 20,
                "بازدهی ۱ روزه پایه (%)": -1.5,
                "بازدهی ۳ روزه پایه (%)": -3.0,
                "بازدهی ۵ روزه پایه (%)": -4.0,
                "فاصله پایه از SMA5 (%)": -1.0,
                "فاصله پایه از SMA20 (%)": -1.0,
                "وضعیت صف پایه": "صف فروش",
                "RSI14 پایه": 40.0,
                "وضعیت RSI پایه": "عادی",
                "پولبک پایه": False,
            },
            # Put واجد شرایط ۲ (اهرم ۵.۰)
            {
                "نماد": "طاهرم_قبول_۲",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار فروش (Put)",
                "قیمت پایانی بازار": 900,
                "قیمت تئوریک BSM": 880,
                "دلتا": -0.30,
                "عمیقاً بی‌ارزش": False,
                "قیمت اعمال": 20000,
                "ارزش معاملات امروز (ریال)": 1_500_000_000,
                "میانگین ارزش ۵ روزه (ریال)": 1_200_000_000,
                "تعداد معاملات امروز": 15,
                "روزهای تا سررسید (DTE)": 16,
                "اهرم": 5.0,
                "حباب خام (%)": 2.2,
                "حباب ریالی": 20,
                "بازدهی ۱ روزه پایه (%)": -1.5,
                "بازدهی ۳ روزه پایه (%)": -3.0,
                "بازدهی ۵ روزه پایه (%)": -4.0,
                "فاصله پایه از SMA5 (%)": -1.0,
                "فاصله پایه از SMA20 (%)": -1.0,
                "وضعیت صف پایه": "صف فروش",
                "RSI14 پایه": 40.0,
                "وضعیت RSI پایه": "عادی",
                "پولبک پایه": False,
            },
            # Put رد شده به خاطر اهرم ۲.۲
            {
                "نماد": "طاهرم_رد_اهرم_پایین",
                "دارایی پایه": "اهرم",
                "نوع قرارداد": "اختیار فروش (Put)",
                "قیمت پایانی بازار": 2800,
                "قیمت تئوریک BSM": 2700,
                "دلتا": -0.70,
                "عمیقاً بی‌ارزش": False,
                "قیمت اعمال": 25000,
                "ارزش معاملات امروز (ریال)": 3_000_000_000,
                "میانگین ارزش ۵ روزه (ریال)": 2_500_000_000,
                "تعداد معاملات امروز": 30,
                "روزهای تا سررسید (DTE)": 16,
                "اهرم": 2.2,
                "حباب خام (%)": 3.7,
                "حباب ریالی": 100,
                "بازدهی ۱ روزه پایه (%)": -1.5,
                "بازدهی ۳ روزه پایه (%)": -3.0,
                "بازدهی ۵ روزه پایه (%)": -4.0,
                "فاصله پایه از SMA5 (%)": -1.0,
                "فاصله پایه از SMA20 (%)": -1.0,
                "وضعیت صف پایه": "صف فروش",
                "RSI14 پایه": 40.0,
                "وضعیت RSI پایه": "عادی",
                "پولبک پایه": False,
            },
        ])

        with self.assertLogs(logger="OptionScanner.UnifiedScoring", level="WARNING") as cm:
            df_c, df_p, conc_info = compute_top_call_and_put(df_v4, config, top_n=10)

        # بررسی هشدار لاگ عمق کم Put (چون فقط ۲ قرارداد واجد شرایط بود و زیر ۵ است)
        self.assertTrue(any("هشدار عمق کم بازار Put" in m for m in cm.output), "باید هشدار عمق کم Put در لاگ ثبت شود")

        # بررسی عدم ورود هیچ قرارداد با اهرم زیر ۳.۰
        self.assertNotIn("ضاهرم_رد_اهرم_پایین", df_c["نماد"].tolist(), "قرارداد Call با اهرم زیر ۳ نباید وارد Top شود")
        self.assertNotIn("طاهرم_رد_اهرم_پایین", df_p["نماد"].tolist(), "قرارداد Put با اهرم زیر ۳ نباید وارد Top شود")

        # بررسی تعداد قراردادهای واجد شرایط
        self.assertEqual(len(df_c), 2, "دقیقاً ۲ نماد Call واجد شرایط بود")
        self.assertEqual(len(df_p), 2, "دقیقاً ۲ نماد Put واجد شرایط بود (بدون پرکردن مصنوعی تا ۱۰)")
        self.assertEqual(conc_info["eligible_calls_count"], 2)
        self.assertEqual(conc_info["eligible_puts_count"], 2)

        # بررسی رتبه صدکی اهرم: اهرم ۶.۵ باید رتبه و امتیاز بالاتری نسبت به اهرم ۳.۵ کسب کند
        c_high_lev = df_c[df_c["نماد"] == "ضاهرم_قبول_اهرم_بالا"].iloc[0]
        c_mid_lev = df_c[df_c["نماد"] == "ضاهرم_قبول_اهرم_متوسط"].iloc[0]
        self.assertGreater(c_high_lev["امتیاز اهرم"], c_mid_lev["امتیاز اهرم"])


if __name__ == "__main__":
    unittest.main()


