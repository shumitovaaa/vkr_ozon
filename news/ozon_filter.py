"""
Фильтр релевантности Ozon по полям title и text (без учёта регистра).

Пустой ``text`` для сопоставления заменяется нормализованным ``title``.
Точка входа пайплайна: :func:`maybe_filter_ozon_after_parse`.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OZON_KEYWORDS_LOWER: Tuple[str, ...] = (
    "ozon",
    "озон",
    "ozon holdings",
    "озон-банк",
    "маркетплейс ozon",
    "ozon bank",
)


def normalize_news_cell(value: object) -> str:
    """Привести ячейку к строке; NaN/пусто → ``\"\"``."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.lower() in ("nan", "none"):
        return ""
    return text


def _body_text_for_match(raw_body: object, title_normalized: str) -> str:
    if raw_body is None or (isinstance(raw_body, float) and np.isnan(raw_body)):
        return title_normalized
    stripped = str(raw_body).strip()
    return stripped if stripped else title_normalized


def _any_keyword_in(title_lower: str, text_lower: str) -> bool:
    return any(
        kw in title_lower or kw in text_lower for kw in OZON_KEYWORDS_LOWER
    )


def haystack_matches_ozon(title: object, body: object) -> bool:
    """True, если ключевое слово Ozon есть в title или в text (text пустой → title)."""
    title_norm = normalize_news_cell(title)
    body_for_match = _body_text_for_match(body, title_norm)
    return _any_keyword_in(title_norm.lower(), body_for_match.lower())


def _match_title_text_row(row: pd.Series) -> bool:
    return haystack_matches_ozon(row.get("title", ""), row.get("text", ""))


def _match_text_only_cell(value: object) -> bool:
    return haystack_matches_ozon("", value if value is not None else "")


def filter_ozon_news_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Оставить строки про Ozon. Исходный фрейм не мутируется."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"ожидался pd.DataFrame, получено {type(df).__name__}")
    if df.empty:
        return df.copy(), 0
    n_before = len(df)
    if "title" in df.columns and "text" in df.columns:
        mask = df.apply(_match_title_text_row, axis=1)
    elif "text" in df.columns:
        mask = df["text"].map(_match_text_only_cell)
    else:
        logger.warning("[OZON-FILTER] Нет колонок title/text — пропуск фильтра")
        return df.copy(), 0
    out = df.loc[mask].copy()
    return out, n_before - len(out)


def maybe_filter_ozon_after_parse(
    df: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    log_tag: str = "[NEWS]",
) -> pd.DataFrame:
    """При ``NEWS_POST_PARSE_OZON_FILTER`` — фильтр Ozon и лог статистики."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"ожидался pd.DataFrame, получено {type(df).__name__}")
    if not isinstance(cfg, Mapping):
        raise TypeError(f"ожидалось отображение конфига, получено {type(cfg).__name__}")
    tag = str(log_tag).strip() if log_tag is not None else ""
    if not tag:
        raise ValueError("log_tag не может быть пустым")

    if not bool(cfg.get("NEWS_POST_PARSE_OZON_FILTER", True)):
        return df.copy()

    filtered, n_removed = filter_ozon_news_df(df)
    logger.info(
        "%s Фильтр Ozon: было %s строк, удалено посторонних %s, осталось %s",
        tag,
        len(df),
        n_removed,
        len(filtered),
    )
    return filtered
