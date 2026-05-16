"""
Формирование матрицы признаков и целевых переменных (multi-horizon).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import DEFAULT_HORIZONS, resolve_news_features_flag

from .returns import compute_log_returns

logger = logging.getLogger(__name__)

FEATURE_COLS: List[str] = [
    "RSI14",
    "STOCHK",
    "BBpctb",
    "BBwidth",
    "price_vs_sma20",
    "MACDnorm",
    "vol_std5",
    "vol_std20",
    "ATRnorm",
    "har_d",
    "har_w",
    "har_m",
    "rv_park5",
    "ret_sq_lag1",
    "sign_lag1",
    "ret_abs_ma5",
    "ret_abs_ma20",
    "OBV_z",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "is_month_end",
    "is_quarter_end",
    "is_friday",
    "is_march",
    "is_july",
    "is_oct",
    "in_crisis",
]

NEWS_FEATURE_COLS: List[str] = [
    "news_count",
    "has_news",
    "sentiment_score",
    "sentiment_trend_3",
    "sentiment_volatility",
]

# NaN допустимы до импутации (дни без новостей); dropna только по остальным колонкам
NEWS_SENTIMENT_NAN_OK: frozenset[str] = frozenset(
    {"sentiment_score", "sentiment_trend_3", "sentiment_volatility"}
)


class FeatureBuilder:
    """Строит X и таргеты reg_h / clf_h по конфигу горизонтов."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.use_news_features = resolve_news_features_flag(cfg)
        cols = list(FEATURE_COLS)
        n_lags = max(0, int(cfg.get("N_LAGS", 0)))
        cols.extend(f"log_ret_lag{k}" for k in range(1, n_lags + 1))
        if self.use_news_features:
            cols = cols + NEWS_FEATURE_COLS
        self.feature_cols = cols

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Числовые признаки: сначала убираются строки с NaN по тех. индикаторам
        и news_count/has_news; затем sentiment_* импутируются медианой по дням с новостями.

        Нормализация — в walk-forward по train-фолду.
        """
        available = [c for c in self.feature_cols if c in df.columns]
        missing = set(self.feature_cols) - set(available)
        if missing:
            logger.warning("[BUILDER] Отсутствуют признаки: %s", missing)

        feat = df[available].copy()
        drop_subset = [
            c for c in available if c not in NEWS_SENTIMENT_NAN_OK
        ]
        before = len(feat)
        feat = feat.dropna(subset=drop_subset)

        # Дни без новостей: заполняем медианой по дням с новостями (не нулём)
        for col in NEWS_SENTIMENT_NAN_OK:
            if col not in feat.columns:
                continue
            if "has_news" in feat.columns:
                med = feat.loc[feat["has_news"] > 0, col].median()
            else:
                med = feat[col].median()
            if pd.isna(med):
                med = 0.0
            feat[col] = feat[col].fillna(med)

        logger.info(
            "[BUILDER] Удалено %s строк с NaN (осталось %s)",
            before - len(feat),
            len(feat),
        )
        return feat

    def build_targets(
        self,
        df: pd.DataFrame,
        horizons: Optional[Sequence[int]] = None,
    ) -> pd.DataFrame:
        """
        Построить регрессионные и классификационные цели (direct multi-step).

        Кумулятивная лог-доходность на h шагов вперёд; clf — знак > 0.
        """
        if horizons is None:
            horizons = self.cfg.get("HORIZONS", list(DEFAULT_HORIZONS))

        log_ret = compute_log_returns(df["CLOSE"])
        tgt = pd.DataFrame(index=df.index)

        for h in horizons:
            tgt[f"reg_{h}"] = log_ret.rolling(int(h)).sum().shift(-int(h))
            tgt[f"clf_{h}"] = (tgt[f"reg_{h}"] > 0).astype(int)

        return tgt
