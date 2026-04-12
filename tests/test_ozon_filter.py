"""Тесты фильтра релевантности Ozon (news/ozon_filter.py)."""

from __future__ import annotations

import pandas as pd
import pytest

from news.ozon_filter import (
    filter_ozon_news_df,
    haystack_matches_ozon,
    maybe_filter_ozon_after_parse,
    normalize_news_cell,
)


def test_normalize_news_cell() -> None:
    assert normalize_news_cell(None) == ""
    assert normalize_news_cell(float("nan")) == ""
    assert normalize_news_cell("  Ozon  ") == "Ozon"
    assert normalize_news_cell("nan") == ""


def test_haystack_matches_ozon() -> None:
    assert haystack_matches_ozon("Новости Ozon", "Текст без ключевого слова") is True
    assert haystack_matches_ozon("Заголовок", "Упоминание озон в тексте") is True
    assert haystack_matches_ozon("Прочее", "Сбер") is False
    assert haystack_matches_ozon("Ozon акции", "") is True


def test_filter_ozon_news_df_title_text() -> None:
    df = pd.DataFrame(
        {
            "title": ["A", "B"],
            "text": ["Ozon рост", "другое"],
        }
    )
    out, removed = filter_ozon_news_df(df)
    assert len(out) == 1
    assert removed == 1
    assert out.iloc[0]["title"] == "A"


def test_filter_ozon_news_df_text_only() -> None:
    df = pd.DataFrame({"text": ["ozon news", "noise"]})
    out, removed = filter_ozon_news_df(df)
    assert len(out) == 1
    assert removed == 1


def test_maybe_filter_respects_cfg() -> None:
    df = pd.DataFrame({"title": ["x"], "text": ["y"]})
    cfg_off = {"NEWS_POST_PARSE_OZON_FILTER": False}
    out = maybe_filter_ozon_after_parse(df, cfg_off, log_tag="[T]")
    assert len(out) == 1

    cfg_on = {"NEWS_POST_PARSE_OZON_FILTER": True}
    out2 = maybe_filter_ozon_after_parse(df, cfg_on, log_tag="[T]")
    assert len(out2) == 0


def test_maybe_filter_requires_log_tag() -> None:
    df = pd.DataFrame({"title": ["Ozon"], "text": ["x"]})
    with pytest.raises(ValueError):
        maybe_filter_ozon_after_parse(df, {"NEWS_POST_PARSE_OZON_FILTER": True}, log_tag="")
