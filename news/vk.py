"""
Загрузка постов стены VK (wall.get): дата, заголовок, текст, ссылка.

После сборки строк применяется фильтр релевантности Ozon (см. ``maybe_filter_ozon_after_parse``).
"""

from __future__ import annotations

import logging
import os
import ssl
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
import requests
from zoneinfo import ZoneInfo

from news.filters import (
    align_since_until_to_calendar,
    matches_keywords,
    parse_since_until,
    price_calendar_from_cfg,
)
from news.ozon_filter import maybe_filter_ozon_after_parse
from news.preprocess import clean_news_text

logger = logging.getLogger(__name__)

MSK = ZoneInfo("Europe/Moscow")
_VK_WALL_GET = "https://api.vk.com/method/wall.get"

_VK_ERR_5_HINT = (
    "Нужен действующий пользовательский access_token (не «Защищённый ключ» приложения и не client_secret). "
    "Создайте приложение на https://dev.vk.com/, затем получите токен через OAuth "
    "(например implicit: response_type=token, scope с offline и wall). "
    "Подставьте полную строку в CFG['VK_ACCESS_TOKEN'] или в переменную окружения VK_ACCESS_TOKEN."
)

_VK_ERR_15_WALL_HINT = (
    " У этого объекта в VK отключена стена или метод wall.get к ней не допускается (часто у ссылок vk.com/im/channels/…). "
    "Используйте публичное сообщество с включённой стеной: задайте VK_GROUP_DOMAIN (короткое имя из vk.com/имя) "
    "и уберите VK_OWNER_ID, либо другой owner_id группы, у которой стена открыта в настройках."
)


def _normalize_vk_token(raw: Any) -> str:
    if raw is None:
        return ""
    t = str(raw).strip()
    if t.startswith("\ufeff"):
        t = t.lstrip("\ufeff")
    t = t.strip('"').strip("'")
    if t.lower().startswith("bearer "):
        t = t[7:].strip()
    return t


def _access_token_from_cfg(cfg: Dict[str, Any]) -> str:
    t = _normalize_vk_token(os.environ.get("VK_ACCESS_TOKEN"))
    if not t:
        t = _normalize_vk_token(cfg.get("VK_ACCESS_TOKEN"))
    return t


def _parse_vk_owner_id(cfg: Dict[str, Any]) -> int | None:
    """Числовой owner_id стены или ссылка вида https://vk.com/im/channels/-230728158."""
    raw = cfg.get("VK_OWNER_ID")
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    s = str(raw).strip()
    if not s:
        return None
    low = s.lower()
    if "im/channels/" in low:
        tail = s.split("im/channels/", 1)[1].split("/")[0].split("?")[0].strip()
        return int(tail)
    return int(s)


_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def _vk_request(
    params: Dict[str, Any],
    *,
    connect_timeout: float,
    read_timeout: float,
    retries: int = 8,
) -> Dict[str, Any]:
    """POST на api.vk.com через requests (TLS/сессии на Windows стабильнее, чем urllib)."""
    headers = {
        "User-Agent": _UA,
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }
    last: Exception | None = None
    timeout = (connect_timeout, read_timeout)
    retry_http = {429, 500, 502, 503, 504}

    for attempt in range(max(1, retries)):
        try:
            r = requests.post(
                _VK_WALL_GET,
                data=params,
                headers=headers,
                timeout=timeout,
            )
            if r.status_code in retry_http and attempt + 1 < retries:
                logger.warning(
                    "[VK] HTTP %s (попытка %s/%s), повтор…",
                    r.status_code,
                    attempt + 1,
                    retries,
                )
                time.sleep(min(45.0, 2.0 ** attempt))
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            body = (
                e.response.text[:500]
                if e.response is not None and e.response.text
                else ""
            )
            if code in retry_http and attempt + 1 < retries:
                logger.warning(
                    "[VK] HTTP %s (попытка %s/%s): %s…",
                    code,
                    attempt + 1,
                    retries,
                    body[:120],
                )
                time.sleep(min(45.0, 2.0 ** attempt))
                continue
            raise RuntimeError(f"VK HTTP {code}: {body}") from e
        except (
            requests.exceptions.RequestException,
            ssl.SSLError,
            TimeoutError,
            OSError,
        ) as e:
            last = e
            logger.warning("[VK] сеть (попытка %s/%s): %s", attempt + 1, retries, e)
            if attempt + 1 < retries:
                time.sleep(min(60.0, 1.5 * (2**attempt)))
                continue
            raise RuntimeError(
                "Нет ответа от api.vk.com (обрыв/TCP 10054 и т.п.). "
                "Попробуйте VPN или другую сеть; отключите прокси антивируса для Python; "
                "увеличьте VK_CONNECT_TIMEOUT_SEC / VK_REQUEST_TIMEOUT_SEC и VK_REQUEST_RETRIES."
            ) from last


def fetch_vk_news(cfg: Dict[str, Any]) -> pd.DataFrame:
    token = _access_token_from_cfg(cfg)
    if not token:
        raise ValueError(
            "Задайте VK_ACCESS_TOKEN: переменная окружения VK_ACCESS_TOKEN или CFG['VK_ACCESS_TOKEN']."
        )
    if len(token) < 32:
        logger.warning(
            "[VK] Токен очень короткий — обычно user access_token длиннее (часто начинается с vk1.a.)."
        )

    owner_id = None
    try:
        owner_id = _parse_vk_owner_id(cfg)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "VK_OWNER_ID: укажите целое число (напр. -230728158) или полную ссылку на канал im/channels/…"
        ) from e

    domain = ""
    if owner_id is None:
        domain = str(cfg.get("VK_GROUP_DOMAIN", "")).strip()
        if domain.startswith("https://vk.com/"):
            domain = domain.replace("https://vk.com/", "").strip("/")
        if domain.startswith("http://vk.com/"):
            domain = domain.replace("http://vk.com/", "").strip("/")
        if domain.startswith("vk.com/"):
            domain = domain.split("/", 1)[-1]
        if not domain:
            raise ValueError(
                "Задайте VK_OWNER_ID (число или ссылка im/channels/…) либо VK_GROUP_DOMAIN (короткое имя сообщества)."
            )
        logger.info("[VK] wall.get: domain=%s", domain)
    else:
        logger.info("[VK] wall.get: owner_id=%s", owner_id)

    api_v = str(cfg.get("VK_API_VERSION", "5.131"))
    connect_timeout = float(cfg.get("VK_CONNECT_TIMEOUT_SEC", 25.0))
    read_timeout = float(cfg.get("VK_REQUEST_TIMEOUT_SEC", 60.0))
    api_retries = int(cfg.get("VK_REQUEST_RETRIES", 8))
    delay = float(cfg.get("VK_REQUEST_DELAY_SEC", 0.35))
    max_rows_n = cfg.get("NEWS_MAX_ARTICLES")
    max_rows = int(max_rows_n) if max_rows_n is not None else None
    max_scan = int(cfg.get("VK_WALL_MAX_POSTS_SCAN", 150_000))

    since, until = parse_since_until(cfg)
    keywords = cfg.get("NEWS_KEYWORDS", ["OZON"])
    if isinstance(keywords, str):
        keywords = [keywords]

    price_calendar = price_calendar_from_cfg(cfg, log_tag="[VK]")
    since, until = align_since_until_to_calendar(since, until, price_calendar, cfg)

    rows: List[Dict[str, Any]] = []
    offset = 0
    scanned = 0
    stop_paging = False

    while scanned < max_scan and not stop_paging:
        params: Dict[str, Any] = {
            "access_token": token,
            "v": api_v,
            "count": 100,
            "offset": offset,
        }
        if owner_id is not None:
            params["owner_id"] = owner_id
        else:
            params["domain"] = domain
        raw = _vk_request(
            params,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retries=api_retries,
        )
        if "error" in raw:
            e = raw["error"]
            code = e.get("error_code")
            msg = e.get("error_msg", "")
            extra = ""
            if code == 5:
                extra = f" {_VK_ERR_5_HINT}"
            elif code == 15 or "wall is disabled" in str(msg).lower():
                extra = f" {_VK_ERR_15_WALL_HINT}"
            raise RuntimeError(f"VK API: {msg} (code {code}).{extra}")
        resp = raw.get("response") or {}
        items = resp.get("items") if isinstance(resp, dict) else None
        if items is None and isinstance(resp, list):
            items = resp
        if not items:
            break

        for post in items:
            scanned += 1
            if scanned > max_scan:
                stop_paging = True
                break

            if post.get("marked_as_ads"):
                continue

            text = (post.get("text") or "").strip()
            if not text:
                continue

            ts = post.get("date")
            if ts is None:
                continue
            dt_utc = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            dt_naive_msk = dt_utc.astimezone(MSK).replace(tzinfo=None)
            msg_msk_date = dt_naive_msk.date()

            if until is not None and msg_msk_date > until:
                continue
            if since is not None and msg_msk_date < since:
                stop_paging = True
                break

            if price_calendar is not None and msg_msk_date not in price_calendar:
                continue

            title = text[:200].replace("\n", " ")
            if not matches_keywords(f"{title}\n{text}", keywords):
                continue

            owner_id = post.get("owner_id")
            post_id = post.get("id")
            if owner_id is None or post_id is None:
                continue
            url = f"https://vk.com/wall{owner_id}_{post_id}"

            rows.append(
                {
                    "date": dt_naive_msk,
                    "title": clean_news_text(title, lowercase=False),
                    "text": clean_news_text(text, lowercase=False),
                    "url": url,
                    "category": "vk",
                }
            )

            if max_rows is not None and len(rows) >= max_rows:
                stop_paging = True
                break

        if stop_paging:
            break
        if len(items) < 100:
            break
        offset += 100
        if delay > 0:
            time.sleep(delay)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    logger.info("[VK] Записей после парсинга и ключевых слов VK: %s (просмотрено постов: %s)", len(df), scanned)
    df = maybe_filter_ozon_after_parse(df, cfg, log_tag="[VK]")
    return df
