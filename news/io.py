"""
Загрузка новостей из CSV: нормализация колонок, очистка текста, фильтр Ozon.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from news.ozon_filter import maybe_filter_ozon_after_parse
from news.preprocess import clean_news_text, combine_headline_body

logger = logging.getLogger(__name__)

_DATE_ALIASES = ("date", "published", "published_at", "pub_date", "datetime", "time")
_TITLE_ALIASES = ("title", "headline", "subject")
_BODY_ALIASES = ("text", "body", "content", "article", "summary")


def load_news_csv(path: str | Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл новостей не найден: {p.resolve()}")

    raw = pd.read_csv(p)
    raw.columns = [str(c).strip() for c in raw.columns]

    date_col = cfg.get("NEWS_DATE_COL")
    if date_col is None:
        date_col = next(
            (c for c in raw.columns if c.lower() in _DATE_ALIASES),
            None,
        )
    if date_col is None or date_col not in raw.columns:
        raise ValueError(
            "Не найдена колонка даты. Задайте NEWS_DATE_COL или используйте одно из имён: "
            + ", ".join(_DATE_ALIASES)
        )

    title_col = cfg.get("NEWS_TITLE_COL")
    if title_col is None:
        title_col = next(
            (c for c in raw.columns if c.lower() in _TITLE_ALIASES),
            None,
        )

    body_col = cfg.get("NEWS_BODY_COL")
    if body_col is None:
        body_col = next(
            (c for c in raw.columns if c.lower() in _BODY_ALIASES),
            None,
        )

    out = pd.DataFrame(index=raw.index)
    out["date"] = pd.to_datetime(raw[date_col], errors="coerce")

    if title_col is None and body_col is None:
        text_candidates = [c for c in raw.columns if c != date_col]
        if not text_candidates:
            raise ValueError("В CSV нет текстовых колонок")
        single = text_candidates[0]
        out["text"] = raw[single].astype(str)
    else:
        titles = raw[title_col].astype(str) if title_col else ""
        bodies = raw[body_col].astype(str) if body_col else ""
        if title_col and body_col:
            out["text"] = [
                combine_headline_body(t, b) for t, b in zip(titles, bodies)
            ]
        elif title_col:
            out["text"] = [combine_headline_body(t, None) for t in titles]
        else:
            out["text"] = [combine_headline_body(None, b) for b in bodies]

    out["text"] = out["text"].map(lambda x: clean_news_text(x, lowercase=False))
    if "sentiment" in raw.columns:
        out["sentiment"] = pd.to_numeric(raw["sentiment"], errors="coerce")

    bad_dates = int(out["date"].isna().sum())
    if bad_dates:
        logger.warning("[NEWS] Строк с некорректной датой (удалены): %s", bad_dates)
    out = out.dropna(subset=["date"])
    out = out[out["text"].str.len() > 0]
    out = maybe_filter_ozon_after_parse(out, cfg, log_tag="[NEWS]")
    if "sentiment" in out.columns:
        out = out.dropna(subset=["sentiment"])
    out = out.sort_values("date").reset_index(drop=True)
    logger.info("[NEWS] Загружено новостей: %s", len(out))
    return out
