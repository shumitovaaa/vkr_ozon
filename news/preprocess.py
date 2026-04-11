"""
Очистка текста новостей и склейка заголовка с телом (clean_news_text, combine_headline_body).
"""

from __future__ import annotations

import html
import re
from typing import Optional

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_news_text(text: Optional[str], *, lowercase: bool = False) -> str:
    if text is None or (isinstance(text, float) and str(text) == "nan"):
        return ""
    s = str(text).strip()
    if not s:
        return ""
    s = html.unescape(s)
    s = _TAG_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    if lowercase:
        s = s.lower()
    return s


def combine_headline_body(title: Optional[str], body: Optional[str]) -> str:
    t = clean_news_text(title, lowercase=False)
    b = clean_news_text(body, lowercase=False)
    if t and b:
        return f"{t}. {b}"
    return t or b
