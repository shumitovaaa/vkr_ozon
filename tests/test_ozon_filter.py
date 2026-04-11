"""Тесты модуля news.ozon_filter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from news.ozon_filter import (
    filter_ozon_news_df,
    haystack_matches_ozon,
    maybe_filter_ozon_after_parse,
    normalize_news_cell,
)


@pytest.mark.parametrize(
    ("title", "body", "expected"),
    [
        ("Доставка OZON", "текст", True),
        ("прочее", "озон банк новости", True),
        ("прочее", "hello world", False),
        ("", "", False),
    ],
)
def test_haystack_matches_ozon(
    title: str,
    body: str,
    expected: bool,
) -> None:
    assert haystack_matches_ozon(title, body) is expected


def test_haystack_empty_body_falls_back_to_title() -> None:
    assert haystack_matches_ozon("OZON акции", "") is True


def test_haystack_nan_body_falls_back_to_title() -> None:
    assert haystack_matches_ozon("озон", np.nan) is True


def test_normalize_news_cell() -> None:
    assert normalize_news_cell(None) == ""
    assert normalize_news_cell(np.nan) == ""
    assert normalize_news_cell("  x  ") == "x"


def test_filter_title_text_columns() -> None:
    df = pd.DataFrame(
        {
            "title": ["a", "b"],
            "text": ["OZON sale", "other"],
        }
    )
    out, removed = filter_ozon_news_df(df)
    assert len(out) == 1
    assert removed == 1
    assert out.iloc[0]["title"] == "a"


def test_filter_text_only_column() -> None:
    df = pd.DataFrame({"text": ["ozon news", "spam"]})
    out, removed = filter_ozon_news_df(df)
    assert len(out) == 1
    assert removed == 1


def test_filter_no_text_columns_returns_copy_unchanged(caplog) -> None:
    import logging

    df = pd.DataFrame({"x": [1]})
    with caplog.at_level(logging.WARNING):
        out, removed = filter_ozon_news_df(df)
    assert removed == 0
    assert "title/text" in caplog.text
    assert len(out) == 1


def test_filter_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        filter_ozon_news_df("not a frame")  # type: ignore[arg-type]


def test_maybe_filter_disabled_returns_copy() -> None:
    df = pd.DataFrame({"text": ["noise"]})
    cfg = {"NEWS_POST_PARSE_OZON_FILTER": False}
    out = maybe_filter_ozon_after_parse(df, cfg)
    assert len(out) == 1
    assert out is not df


def test_maybe_filter_applied() -> None:
    df = pd.DataFrame({"text": ["ozon ok", "bad"]})
    cfg: dict = {}
    out = maybe_filter_ozon_after_parse(df, cfg, log_tag="[T]")
    assert len(out) == 1


def test_maybe_filter_validates_cfg() -> None:
    df = pd.DataFrame({"text": ["x"]})
    with pytest.raises(TypeError, match="отображение"):
        maybe_filter_ozon_after_parse(df, [])  # type: ignore[arg-type]


def test_maybe_filter_validates_log_tag() -> None:
    df = pd.DataFrame({"text": ["ozon"]})
    with pytest.raises(ValueError, match="log_tag"):
        maybe_filter_ozon_after_parse(df, {}, log_tag="   ")
