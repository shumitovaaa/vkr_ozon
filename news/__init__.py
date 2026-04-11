"""news — загрузка новостей (VK, CSV), тональность и дневные признаки."""

from __future__ import annotations

from news.merge import attach_news_sentiment_features
from news.vk import fetch_vk_news

__all__ = ["attach_news_sentiment_features", "fetch_vk_news"]
