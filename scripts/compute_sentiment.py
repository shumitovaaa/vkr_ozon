#!/usr/bin/env python3
"""
Часть 1: фильтрация новостей по релевантности Ozon (поля title и text, регистронезависимо).
Часть 2 (опционально): колонка sentiment через ensure_sentiment_column (или только фильтр).

Примеры:
  python scripts/compute_sentiment.py --input data/news.csv
  python scripts/compute_sentiment.py --input data/news.csv --no-inference
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from news.ozon_filter import filter_ozon_news_df, normalize_news_cell
from news.preprocess import combine_headline_body
from news.sentiment import ensure_sentiment_column

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def build_model_text_column(df: pd.DataFrame) -> pd.Series:
    """Текст как в pipeline: combine_headline_body(title, text)."""
    n = len(df)
    if "title" not in df.columns or "text" not in df.columns:
        raise ValueError("Нужны колонки title и text")
    out: list[str] = []
    for i in range(n):
        t = normalize_news_cell(df["title"].iloc[i])
        b = df["text"].iloc[i]
        if b is None or (isinstance(b, float) and np.isnan(b)):
            b = None
        else:
            bs = str(b).strip()
            b = bs if bs else None
        out.append(combine_headline_body(t or None, b))
    return pd.Series(out, index=df.index)


def _read_news_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    cmap = {str(c).strip().lower(): str(c).strip() for c in raw.columns}

    def req(name: str) -> str:
        if name.lower() not in cmap:
            raise SystemExit(
                f"В CSV нет колонки «{name}». Доступны: {list(raw.columns)}"
            )
        return cmap[name.lower()]

    out = pd.DataFrame(
        {
            "date": raw[req("date")],
            "title": raw[req("title")],
            "text": raw[req("text")],
        }
    )
    if "url" in cmap:
        out["url"] = raw[cmap["url"]]
    if "category" in cmap:
        out["category"] = raw[cmap["category"]]
    if "sentiment" in cmap:
        out["sentiment"] = raw[cmap["sentiment"]]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Фильтр Ozon + опционально sentiment")
    p.add_argument("--input", type=Path, default=_ROOT / "data" / "news.csv")
    p.add_argument("--out-filtered", type=Path, default=_ROOT / "data" / "news_filtered.csv")
    p.add_argument(
        "--out-sentiment",
        type=Path,
        default=_ROOT / "data" / "news_filtered_sentiment.csv",
        help="Файл для NEWS_CSV_PATH (с колонкой sentiment)",
    )
    p.add_argument(
        "--no-inference",
        action="store_true",
        help="Не вызывать модель: взять sentiment из входного CSV или NaN",
    )
    args = p.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Нет файла: {args.input}")

    df = _read_news_csv(args.input)
    filtered, n_removed = filter_ozon_news_df(df)

    args.out_filtered.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.out_filtered, index=False, encoding="utf-8")
    logger.info(
        "Фильтрация Ozon: было %s строк, осталось %s, удалено %s → %s",
        len(df),
        len(filtered),
        n_removed,
        args.out_filtered.resolve(),
    )

    work = filtered.copy()
    model_text = build_model_text_column(work)

    if args.no_inference:
        if "sentiment" not in work.columns:
            work["sentiment"] = np.nan
            logger.warning(
                "--no-inference: колонка sentiment пустая (заполните вручную или уберите флаг)"
            )
        else:
            work["sentiment"] = pd.to_numeric(work["sentiment"], errors="coerce")
    else:
        has_full = "sentiment" in work.columns and bool(work["sentiment"].notna().all())
        if has_full:
            logger.info("Колонка sentiment уже полностью заполнена — пропуск инференса")
            work["sentiment"] = pd.to_numeric(work["sentiment"], errors="coerce")
        else:
            from config import CFG

            mdf = pd.DataFrame({"text": model_text})
            mdf = ensure_sentiment_column(mdf, "text", CFG)
            work["sentiment"] = mdf["sentiment"].to_numpy()
            logger.info("Инференс тональности: %s строк", len(work))

    args.out_sentiment.parent.mkdir(parents=True, exist_ok=True)
    work.to_csv(args.out_sentiment, index=False, encoding="utf-8")
    logger.info("Сохранено: %s", args.out_sentiment.resolve())


if __name__ == "__main__":
    main()
