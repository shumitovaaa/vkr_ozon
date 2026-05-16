from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from config import resolve_news_features_flag
from news.aggregate import daily_sentiment_features
from news.io import load_news_csv
from news.sentiment import ensure_sentiment_column

logger = logging.getLogger(__name__)

NEWS_FEATURE_COLS = [
    "news_count",
    "has_news",
    "sentiment_score",
    "sentiment_trend_3",
    "sentiment_volatility",
]


def _ensure_zero_news_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["news_count"] = 0
    out["has_news"] = 0
    nan_col = np.nan * np.ones(len(out), dtype=np.float64)
    out["sentiment_score"] = nan_col
    out["sentiment_trend_3"] = nan_col
    out["sentiment_volatility"] = nan_col
    return out


def attach_news_sentiment_features(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Подключить дневные новостные признаки к ``df`` (по торговому индексу), если
    ``USE_NEWS_FEATURES``/``USE_NEWS_SENTIMENT`` включён. В режиме baseline (False)
    функция возвращает ``df`` без новостных колонок — это и есть сценарий
    «без новостей» в A/B-сравнении.
    """
    if not resolve_news_features_flag(cfg):
        return df

    path = cfg.get("NEWS_CSV_PATH")
    if not path:
        logger.warning("[NEWS] NEWS_CSV_PATH не задан")
        return _ensure_zero_news_features(df)

    path = Path(path)
    if not path.exists():
        logger.warning("[NEWS] Нет файла: %s", path)
        return _ensure_zero_news_features(df)

    news = load_news_csv(path, cfg)
    if news.empty:
        logger.warning("[NEWS] Пустой CSV")
        return _ensure_zero_news_features(df)

    news = ensure_sentiment_column(news, "text", cfg)
    daily = daily_sentiment_features(news, df.index)

    out = df.copy()
    for col in daily.columns:
        out[col] = daily[col].values

    return out
