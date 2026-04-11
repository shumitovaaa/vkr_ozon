"""
Агрегация тональности новостей по торговым дням (daily_sentiment_features).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def daily_sentiment_features(
    news: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if "sentiment" not in news.columns:
        raise ValueError("Ожидается колонка sentiment в таблице новостей")

    nd = news.copy()
    nd["day"] = pd.to_datetime(nd["date"]).dt.normalize()
    g = nd.groupby("day", as_index=True)["sentiment"].agg(["mean", "count"])
    g = g.rename(columns={"mean": "sentiment_score", "count": "news_count"})

    norm_index = pd.DatetimeIndex(trading_index).normalize()
    aligned = g.reindex(norm_index)

    feat = pd.DataFrame(index=trading_index)
    # Дни без новостей — NaN (не путать с нейтральной меткой RuBERT = 0)
    feat["sentiment_score"] = aligned["sentiment_score"].to_numpy()
    feat["news_count"] = aligned["news_count"].fillna(0.0).to_numpy().astype(np.int64)
    feat["has_news"] = (feat["news_count"] > 0).astype(np.int64)

    ss = feat["sentiment_score"]
    feat["sentiment_trend_3"] = ss.rolling(window=3, min_periods=1).mean()
    feat["sentiment_volatility"] = ss.rolling(window=5, min_periods=1).std()

    logger.info(
        "[NEWS] Дневные признаки: дней с новостями=%s из %s",
        int(feat["has_news"].sum()),
        len(feat),
    )
    return feat
