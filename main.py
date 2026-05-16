"""
Точка входа: полный эксперимент, smoke-тест или A/B-сравнение «без новостей vs с новостями».

Режимы:

* по умолчанию — один прогон ``run_experiment`` (флаг ``USE_NEWS_FEATURES`` берётся из ``CFG``);
* ``--ablation`` — два прогона ``run_experiment`` (baseline + news-enhanced) и сводные
  таблицы в ``out_dir`` (см. ``pipeline.run_ab_comparison``);
* ``--no-news`` / ``--with-news`` — явное принуждение режима в одиночном прогоне;
* ``--smoke`` — лёгкая проверка окружения (без ML/ARIMA).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from config import CFG, DEFAULT_FILE_PATH, DEFAULT_OUT_DIR
from pipeline import run_ab_comparison, run_experiment

SMOKE_SAMPLE_ROWS = 50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def smoke_test(filepath: str) -> None:
    """
    Проверка на первых 50 строках: загрузка, валидация, признаки, таргеты.

    ML не запускается.
    """
    from data.loader import load_data
    from data.validator import validate_data
    from features.builder import FeatureBuilder
    from features.indicators import compute_indicators

    raw = load_data(filepath).head(SMOKE_SAMPLE_ROWS)
    raw = validate_data(raw, CFG)
    df = compute_indicators(raw, CFG)
    fb = FeatureBuilder(CFG)
    X = fb.build_features(df)
    tgt = fb.build_targets(df, horizons=(1,))
    logger.info(
        "[SMOKE] OK — df=%s, X=%s, reg_1=%s",
        df.shape,
        X.shape,
        tgt["reg_1"].dropna().shape,
    )


def run(
    data_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    *,
    use_news_features: Optional[bool] = None,
) -> None:
    """Запуск run_experiment с путями из аргументов или CFG (опц. переопределение режима)."""
    path = data_path or CFG.get("FILE_PATH", DEFAULT_FILE_PATH)
    out = out_dir or CFG.get("OUT_DIR", DEFAULT_OUT_DIR)
    run_experiment(
        cfg=CFG,
        data_path=path,
        out_dir=out,
        use_news_features=use_news_features,
    )


def run_ablation(data_path: Optional[str] = None, out_dir: Optional[str] = None) -> None:
    """A/B сравнение «без новостей vs с новостями» (см. ``pipeline.run_ab_comparison``)."""
    path = data_path or CFG.get("FILE_PATH", DEFAULT_FILE_PATH)
    out = out_dir or CFG.get("OUT_DIR", DEFAULT_OUT_DIR)
    run_ab_comparison(cfg=CFG, data_path=path, out_dir=out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OZON TA Pipeline")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "A/B эксперимент: запустить пайплайн дважды (без новостей и с новостями) "
            "и сохранить сводные таблицы Δ-метрик."
        ),
    )
    news_group = parser.add_mutually_exclusive_group()
    news_group.add_argument(
        "--with-news",
        dest="use_news",
        action="store_true",
        default=None,
        help="Принудительно включить новостные признаки в одиночном прогоне.",
    )
    news_group.add_argument(
        "--no-news",
        dest="use_news",
        action="store_false",
        default=None,
        help="Принудительно выключить новостные признаки в одиночном прогоне (baseline).",
    )
    args = parser.parse_args()
    default_data = CFG.get("FILE_PATH", DEFAULT_FILE_PATH)

    if args.smoke:
        smoke_test(args.data or default_data)
    elif args.ablation:
        run_ablation(data_path=args.data, out_dir=args.out)
    else:
        run(
            data_path=args.data,
            out_dir=args.out,
            use_news_features=args.use_news,
        )
