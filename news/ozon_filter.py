"""
Фильтр релевантности Ozon по полям ``title`` и ``text`` (без учёта регистра).

Пустой ``text`` для сопоставления заменяется нормализованным ``title``.
Точка входа пайплайна: :func:`maybe_filter_ozon_after_parse`.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Нижний регистр — шаблоны для подстрокового поиска в lower()-строках
OZON_KEYWORDS_LOWER: Tuple[str, ...] = (
    "ozon",
    "озон",
    "ozon holdings",
    "озон-банк",
    "маркетплейс ozon",
    "ozon bank",
)


def normalize_news_cell(value: object) -> str:
    """
    Привести ячейку CSV/DataFrame к строке для сравнения.

    Args:
        value: Скаляр из pandas (в т.ч. NaN) или None.

    Returns:
        Строка без лишних пробелов; для «пустых» значений — ``""``.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if text.lower() in ("nan", "none"):
        return ""
    return text


def _body_text_for_match(raw_body: object, title_normalized: str) -> str:
    """Выбрать текст «тела» для поиска ключевых слов (fallback — title)."""
    if raw_body is None or (isinstance(raw_body, float) and np.isnan(raw_body)):
        return title_normalized
    stripped = str(raw_body).strip()
    return stripped if stripped else title_normalized


def _any_keyword_in(title_lower: str, text_lower: str) -> bool:
    """True, если хотя бы одно ключевое слово встречается в title или в text."""
    return any(
        kw in title_lower or kw in text_lower
        for kw in OZON_KEYWORDS_LOWER
    )


def haystack_matches_ozon(title: object, body: object) -> bool:
    """
    Проверить наличие ключевых слов Ozon в заголовке и/или тексте.

    Args:
        title: Заголовок (строка или значение из ячейки).
        body: Текст; если пустой/NaN, для поиска по «телу» используется title.

    Returns:
        True, если найдено совпадение с одним из шаблонов (регистр не важен).
    """
    title_norm = normalize_news_cell(title)
    body_for_match = _body_text_for_match(body, title_norm)
    title_lower = title_norm.lower()
    text_lower = body_for_match.lower()
    return _any_keyword_in(title_lower, text_lower)


def _match_title_text_row(row: pd.Series) -> bool:
    """Строка с колонками title и text соответствует фильтру Ozon."""
    return haystack_matches_ozon(row.get("title", ""), row.get("text", ""))


def _match_text_only_cell(value: object) -> bool:
    """Одна колонка text: ключевые слова только в ней (title не задан)."""
    return haystack_matches_ozon("", value if value is not None else "")


def filter_ozon_news_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Оставить строки, релевантные Ozon по ключевым словам.

    Ожидаются либо колонки ``title`` и ``text``, либо только ``text``.
    Исходный фрейм не мутируется.

    Args:
        df: Таблица новостей.

    Returns:
        Кортеж ``(отфильтрованный_датафрейм, число_удалённых_строк)``.

    Raises:
        TypeError: Если ``df`` не :class:`pandas.DataFrame`.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"ожидался pd.DataFrame, получено {type(df).__name__}"
        )
    if df.empty:
        return df.copy(), 0

    n_before = len(df)
    if "title" in df.columns and "text" in df.columns:
        mask = df.apply(_match_title_text_row, axis=1)
    elif "text" in df.columns:
        mask = df["text"].map(_match_text_only_cell)
    else:
        logger.warning(
            "[OZON-FILTER] Нет колонок title/text — пропуск фильтра"
        )
        return df.copy(), 0

    out = df.loc[mask].copy()
    n_removed = n_before - len(out)
    return out, n_removed


def maybe_filter_ozon_after_parse(
    df: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    log_tag: str = "[NEWS]",
) -> pd.DataFrame:
    """
    Применить фильтр Ozon, если это задано в конфигурации.

    Ключ ``NEWS_POST_PARSE_OZON_FILTER``: при ``False`` возвращается копия
    входного фрейма без фильтрации.

    Args:
        df: Данные после парсинга или загрузки.
        cfg: Словарь настроек (например, ``config.CFG``).
        log_tag: Префикс для записей лога (контекст: VK, NEWS и т.д.).

    Returns:
        Отфильтрованный или исходный (при отключённом фильтре) DataFrame.

    Raises:
        TypeError: Если ``df`` не DataFrame или ``cfg`` не отображение.
        ValueError: Если ``log_tag`` после обрезки пробелов пустой.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"ожидался pd.DataFrame, получено {type(df).__name__}"
        )
    if not isinstance(cfg, Mapping):
        raise TypeError(
            f"ожидалось отображение конфига, получено {type(cfg).__name__}"
        )
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
