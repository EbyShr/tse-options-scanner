"""
ماژول بارگذاری و اعتبارسنجی تنظیمات اسکنر آپشن
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ValuationConfig:
    risk_free_rate: float = 0.28
    historical_volatility_window: int = 30
    dividend_yield: float = 0.0


@dataclass
class LiquidityFilterConfig:
    min_trade_value_rials: float = 500_000_000.0
    min_trade_count: int = 5
    lookback_days_avg: int = 5
    low_liquidity_label: str = "کم‌عمق / ریسک نقدشوندگی بالا"
    normal_liquidity_label: str = "نقدشونده / عادی"


@dataclass
class ScoringWeightsConfig:
    weight_bubble: float = 0.40
    weight_liquidity: float = 0.35
    weight_leverage: float = 0.25
    illiquid_rank_penalty: float = 0.50


@dataclass
class UnifiedWeightsConfig:
    underlying_readiness: float = 0.25
    relative_value: float = 0.20
    leverage: float = 0.15
    combined_liquidity: float = 0.25
    dte_suitability: float = 0.15


@dataclass
class LiquidityBreakdownConfig:
    structural_weight: float = 0.15
    spike_weight: float = 0.10


@dataclass
class OptionTypeHardFilterConfig:
    min_trade_value_rials: float = 5_000_000_000.0
    min_trade_count: int = 30
    min_leverage: float = 3.0


@dataclass
class HardFilterConfig:
    min_dte: int = 3
    min_trade_value_rials: float = 5_000_000_000.0
    min_trade_count: int = 30
    min_leverage: float = 3.0
    min_leverage_call: float = 3.0
    min_leverage_put: float = 3.0
    call: OptionTypeHardFilterConfig = field(
        default_factory=lambda: OptionTypeHardFilterConfig(min_trade_value_rials=5_000_000_000.0, min_trade_count=30, min_leverage=3.0)
    )
    put: OptionTypeHardFilterConfig = field(
        default_factory=lambda: OptionTypeHardFilterConfig(min_trade_value_rials=1_000_000_000.0, min_trade_count=10, min_leverage=3.0)
    )


@dataclass
class AntiChasingConfig:
    penalty_factor: float = 0.70
    chasing_safety_ratio: float = 0.90
    daily_bands: Dict[str, float] = field(
        default_factory=lambda: {"leveraged": 0.04, "regular": 0.03}
    )
    leveraged_symbols: List[str] = field(
        default_factory=lambda: ["اهرم", "توان", "موج", "اطلس", "جهش", "شتاب", "نارنج", "بیدار"]
    )
    return_3d_threshold_pct: float = 15.0
    fallback_return_1d_threshold_pct: float = 5.0
    threshold_pct: float = 15.0


@dataclass
class DeepOtmCapConfig:
    min_bsm_price: float = 10.0
    min_delta: float = 0.10


@dataclass
class DteCurveConfig:
    optimal_min: int = 10
    optimal_max: int = 40


@dataclass
class UnifiedScoringConfig:
    weights: UnifiedWeightsConfig = field(default_factory=UnifiedWeightsConfig)
    liquidity_breakdown: LiquidityBreakdownConfig = field(default_factory=LiquidityBreakdownConfig)
    hard_filter: HardFilterConfig = field(default_factory=HardFilterConfig)
    anti_chasing: AntiChasingConfig = field(default_factory=AntiChasingConfig)
    deep_otm_cap: DeepOtmCapConfig = field(default_factory=DeepOtmCapConfig)
    dte_curve: DteCurveConfig = field(default_factory=DteCurveConfig)
    max_per_underlying: int = 3
    max_same_underlying_alert: int = 3


@dataclass
class TopEntryConfig:
    min_trade_value_rials: float = 5_000_000_000.0
    min_dte: int = 3
    min_trade_count: int = 30
    weight_liquidity: float = 0.30
    weight_relative_bubble: float = 0.30
    weight_underlying_readiness: float = 0.25
    weight_dte_suitability: float = 0.15
    optimal_dte_min: int = 10
    optimal_dte_max: int = 40
    max_per_underlying: int = 3
    max_same_underlying_alert: int = 3


@dataclass
class OutputConfig:
    directory: str = "outputs"
    filename_prefix: str = "options_scan"
    export_csv: bool = True
    export_top_choices_csv: bool = True


@dataclass
class NetworkConfig:
    request_timeout: int = 12
    max_retries: int = 3
    concurrency_workers: int = 10


@dataclass
class AppConfig:
    target_underlying_symbols: List[str] = field(
        default_factory=lambda: [
            "وبملت", "خودرو", "شستا", "خساپا", "وتجارت", "فملی",
            "اخابر", "وبصادر", "ذوب", "شپنا", "خبهمن", "فرابورس",
            "تاصیکو", "بساما", "اهرم", "فزر", "کاریس", "دارونو",
            "موج", "توان", "اطلس", "جوانه کوچک", "همتراز", "طعام"
        ]
    )
    valuation: ValuationConfig = field(default_factory=ValuationConfig)
    liquidity_filter: LiquidityFilterConfig = field(default_factory=LiquidityFilterConfig)
    scoring_weights: ScoringWeightsConfig = field(default_factory=ScoringWeightsConfig)
    unified_scoring: UnifiedScoringConfig = field(default_factory=UnifiedScoringConfig)
    top_entry: TopEntryConfig = field(default_factory=TopEntryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """بارگذاری فایل تنظیمات و ایجاد آبجکت AppConfig با مقادیر پیش‌فرض امن"""
    if not os.path.exists(config_path):
        print(f"هشدار: فایل کانفیگ {config_path} یافت نشد، از مقادیر پیش‌فرض استفاده می‌شود.")
        return AppConfig()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        targets = data.get(
            "target_underlying_symbols",
            [
                "وبملت", "خودرو", "شستا", "خساپا", "وتجارت", "فملی",
                "اخابر", "وبصادر", "ذوب", "شپنا", "خبهمن", "فرابورس",
                "تاصیکو", "بساما", "اهرم", "فزر", "کاریس", "دارونو",
                "موج", "توان", "اطلس", "جوانه کوچک", "همتراز", "طعام"
            ],
        )

        val_data = data.get("valuation", {})
        valuation = ValuationConfig(
            risk_free_rate=float(val_data.get("risk_free_rate", 0.28)),
            historical_volatility_window=int(val_data.get("historical_volatility_window", 30)),
            dividend_yield=float(val_data.get("dividend_yield", 0.0)),
        )

        liq_data = data.get("liquidity_filter", {})
        liquidity = LiquidityFilterConfig(
            min_trade_value_rials=float(liq_data.get("min_trade_value_rials", 500_000_000.0)),
            min_trade_count=int(liq_data.get("min_trade_count", 5)),
            lookback_days_avg=int(liq_data.get("lookback_days_avg", 5)),
            low_liquidity_label=str(liq_data.get("low_liquidity_label", "کم‌عمق / ریسک نقدشوندگی بالا")),
            normal_liquidity_label=str(liq_data.get("normal_liquidity_label", "نقدشونده / عادی")),
        )

        score_data = data.get("scoring_weights", {})
        scoring = ScoringWeightsConfig(
            weight_bubble=float(score_data.get("weight_bubble", 0.40)),
            weight_liquidity=float(score_data.get("weight_liquidity", 0.35)),
            weight_leverage=float(score_data.get("weight_leverage", 0.25)),
            illiquid_rank_penalty=float(score_data.get("illiquid_rank_penalty", 0.50)),
        )

        # الگوریتم جدید امتیازدهی واحد
        uni_data = data.get("unified_scoring", {})
        u_weights_data = uni_data.get("weights", {})
        u_liq_data = uni_data.get("liquidity_breakdown", {})
        u_hard_data = uni_data.get("hard_filter", {})
        u_anti_data = uni_data.get("anti_chasing", {})
        u_deep_data = uni_data.get("deep_otm_cap", {})
        u_dte_data = uni_data.get("dte_curve", {})

        call_hard_data = u_hard_data.get("call", {})
        put_hard_data = u_hard_data.get("put", {})
        call_min_val = float(call_hard_data.get("min_trade_value_rials", u_hard_data.get("min_trade_value_rials", 5_000_000_000.0)))
        call_min_trades = int(call_hard_data.get("min_trade_count", u_hard_data.get("min_trade_count", 30)))
        call_min_lev = float(call_hard_data.get("min_leverage", u_hard_data.get("min_leverage_call", u_hard_data.get("min_leverage", 3.0))))
        put_min_val = float(put_hard_data.get("min_trade_value_rials", 1_000_000_000.0))
        put_min_trades = int(put_hard_data.get("min_trade_count", 10))
        put_min_lev = float(put_hard_data.get("min_leverage", u_hard_data.get("min_leverage_put", u_hard_data.get("min_leverage", 3.0))))

        ret_3d_th = float(u_anti_data.get("return_3d_threshold_pct", u_anti_data.get("threshold_pct", 15.0)))
        ret_1d_th = float(u_anti_data.get("fallback_return_1d_threshold_pct", 5.0))
        penalty_f = float(u_anti_data.get("penalty_factor", 0.70))
        safety_ratio = float(u_anti_data.get("chasing_safety_ratio", 0.90))
        daily_bands = u_anti_data.get("daily_bands", {"leveraged": 0.04, "regular": 0.03})
        lev_symbols = u_anti_data.get(
            "leveraged_symbols",
            ["اهرم", "توان", "موج", "اطلس", "جهش", "شتاب", "نارنج", "بیدار"],
        )

        unified = UnifiedScoringConfig(
            weights=UnifiedWeightsConfig(
                underlying_readiness=float(u_weights_data.get("underlying_readiness", 0.25)),
                relative_value=float(u_weights_data.get("relative_value", 0.20)),
                leverage=float(u_weights_data.get("leverage", 0.15)),
                combined_liquidity=float(u_weights_data.get("combined_liquidity", 0.25)),
                dte_suitability=float(u_weights_data.get("dte_suitability", 0.15)),
            ),
            liquidity_breakdown=LiquidityBreakdownConfig(
                structural_weight=float(u_liq_data.get("structural_weight", 0.15)),
                spike_weight=float(u_liq_data.get("spike_weight", 0.10)),
            ),
            hard_filter=HardFilterConfig(
                min_dte=int(u_hard_data.get("min_dte", 3)),
                min_trade_value_rials=call_min_val,
                min_trade_count=call_min_trades,
                min_leverage=call_min_lev,
                min_leverage_call=call_min_lev,
                min_leverage_put=put_min_lev,
                call=OptionTypeHardFilterConfig(
                    min_trade_value_rials=call_min_val,
                    min_trade_count=call_min_trades,
                    min_leverage=call_min_lev,
                ),
                put=OptionTypeHardFilterConfig(
                    min_trade_value_rials=put_min_val,
                    min_trade_count=put_min_trades,
                    min_leverage=put_min_lev,
                ),
            ),
            anti_chasing=AntiChasingConfig(
                penalty_factor=penalty_f,
                chasing_safety_ratio=safety_ratio,
                daily_bands=daily_bands,
                leveraged_symbols=lev_symbols,
                return_3d_threshold_pct=ret_3d_th,
                fallback_return_1d_threshold_pct=ret_1d_th,
                threshold_pct=ret_3d_th,
            ),
            deep_otm_cap=DeepOtmCapConfig(
                min_bsm_price=float(u_deep_data.get("min_bsm_price", 10.0)),
                min_delta=float(u_deep_data.get("min_delta", 0.10)),
            ),
            dte_curve=DteCurveConfig(
                optimal_min=int(u_dte_data.get("optimal_min", 10)),
                optimal_max=int(u_dte_data.get("optimal_max", 40)),
            ),
            max_per_underlying=int(uni_data.get("max_per_underlying", 3)),
            max_same_underlying_alert=int(uni_data.get("max_same_underlying_alert", 3)),
        )

        # سازگاری با top_entry
        top_entry = TopEntryConfig(
            min_trade_value_rials=unified.hard_filter.min_trade_value_rials,
            min_dte=unified.hard_filter.min_dte,
            min_trade_count=unified.hard_filter.min_trade_count,
            weight_liquidity=unified.weights.combined_liquidity,
            weight_relative_bubble=unified.weights.relative_value,
            weight_underlying_readiness=unified.weights.underlying_readiness,
            weight_dte_suitability=unified.weights.dte_suitability,
            optimal_dte_min=unified.dte_curve.optimal_min,
            optimal_dte_max=unified.dte_curve.optimal_max,
            max_per_underlying=unified.max_per_underlying,
            max_same_underlying_alert=unified.max_same_underlying_alert,
        )

        out_data = data.get("output", {})
        output = OutputConfig(
            directory=str(out_data.get("directory", "outputs")),
            filename_prefix=str(out_data.get("filename_prefix", "options_scan")),
            export_csv=bool(out_data.get("export_csv", True)),
            export_top_choices_csv=bool(out_data.get("export_top_choices_csv", True)),
        )

        net_data = data.get("network", {})
        network = NetworkConfig(
            request_timeout=int(net_data.get("request_timeout", 12)),
            max_retries=int(net_data.get("max_retries", 3)),
            concurrency_workers=int(net_data.get("concurrency_workers", 10)),
        )

        return AppConfig(
            target_underlying_symbols=targets,
            valuation=valuation,
            liquidity_filter=liquidity,
            scoring_weights=scoring,
            unified_scoring=unified,
            top_entry=top_entry,
            output=output,
            network=network,
        )

    except Exception as e:
        print(f"خطا در خواندن کانفیگ ({e})، از مقادیر پیش‌فرض استفاده می‌شود.")
        return AppConfig()
