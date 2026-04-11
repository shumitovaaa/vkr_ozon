from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CFG
from news.vk import fetch_vk_news

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Новости (VK) → CSV")
    parser.add_argument("--out", type=str, default="data/news.csv", help="Выходной CSV")
    parser.add_argument("--limit", type=int, default=None, help="Макс. строк после фильтров")
    parser.add_argument("--since", type=str, default=None, help="С даты YYYY-MM-DD")
    parser.add_argument("--until", type=str, default=None, help="По дату YYYY-MM-DD")
    parser.add_argument("--prices", type=str, default=None, help="CSV котировок для календаря дат")
    parser.add_argument("--all-dates", action="store_true", help="Не фильтровать по дням из CSV цен")
    parser.add_argument("--keyword", type=str, action="append", default=None, help="Ключевое слово (повторяемо)")
    parser.add_argument(
        "--vk-domain",
        type=str,
        default=None,
        help="Короткое имя сообщества (альтернатива owner_id)",
    )
    parser.add_argument(
        "--vk-owner-id",
        type=str,
        default=None,
        help="owner_id стены: число -230728158 или ссылка vk.com/im/channels/-230728158",
    )
    args = parser.parse_args()

    cfg = dict(CFG)
    if args.limit is not None:
        cfg["NEWS_MAX_ARTICLES"] = args.limit
    if args.since:
        cfg["NEWS_SINCE"] = args.since
    if args.until:
        cfg["NEWS_UNTIL"] = args.until
    if args.all_dates:
        cfg["NEWS_ONLY_PRICE_DATES"] = False
    if args.prices:
        cfg["NEWS_PRICE_CALENDAR_PATH"] = args.prices
        cfg["NEWS_ONLY_PRICE_DATES"] = True
    if args.keyword:
        cfg["NEWS_KEYWORDS"] = args.keyword
    if args.vk_domain:
        cfg["VK_GROUP_DOMAIN"] = args.vk_domain
    if args.vk_owner_id:
        cfg["VK_OWNER_ID"] = args.vk_owner_id

    df = fetch_vk_news(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    logger.info("Сохранено: %s (%s строк)", out.resolve(), len(df))


if __name__ == "__main__":
    main()
