from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set

logger = logging.getLogger(__name__)


def matches_keywords(text: str, keywords: Optional[Sequence[str]]) -> bool:
    if not keywords:
        return True
    t = text.lower()
    for kw in keywords:
        if kw.lower() in t:
            return True
    return False


def parse_since_until(cfg: Dict[str, Any]) -> tuple[Optional[date], Optional[date]]:
    def _one(key: str) -> Optional[date]:
        v = cfg.get(key)
        if not v:
            return None
        return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()

    return _one("NEWS_SINCE"), _one("NEWS_UNTIL")


def price_calendar_from_cfg(
    cfg: Dict[str, Any],
    *,
    log_tag: str = "[NEWS]",
) -> Optional[Set[date]]:
    if not bool(cfg.get("NEWS_ONLY_PRICE_DATES", True)):
        return None
    cal_path = cfg.get("NEWS_PRICE_CALENDAR_PATH") or cfg.get("FILE_PATH")
    if not cal_path:
        return None
    from data.loader import load_trading_dates

    p = Path(str(cal_path))
    if not p.is_file():
        logger.warning("%s Календарь цен не найден: %s", log_tag, p.resolve())
        return None
    cal = load_trading_dates(p)
    logger.info("%s Календарь цен: %s дней (%s)", log_tag, len(cal), p)
    return cal


def align_since_until_to_calendar(
    since: Optional[date],
    until: Optional[date],
    price_calendar: Optional[Set[date]],
    cfg: Dict[str, Any],
) -> tuple[Optional[date], Optional[date]]:
    """
    Если в конфиге включено и загружен календарь цен — подставить границы min/max дат из файла,
    когда NEWS_SINCE / NEWS_UNTIL не заданы.
    «Только дни из файла котировок» — NEWS_ONLY_PRICE_DATES.
    """
    if not price_calendar:
        return since, until
    if not bool(cfg.get("NEWS_ALIGN_BOUNDS_TO_CALENDAR", True)):
        return since, until
    had_open = since is None or until is None
    if since is None:
        since = min(price_calendar)
    if until is None:
        until = max(price_calendar)
    if had_open:
        logger.info("[NEWS] Окно дат по календарю цен: since=%s until=%s", since, until)
    return since, until
