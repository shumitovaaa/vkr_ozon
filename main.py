"""
Точка входа: полный эксперимент или smoke-тест.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from config import CFG
from pipeline import run_experiment

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

    raw = load_data(filepath).head(50)
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


def run(data_path: Optional[str] = None, out_dir: Optional[str] = None) -> None:
    """Запуск run_experiment с путями из аргументов или CFG."""
    path = data_path or CFG.get("FILE_PATH", "OZON_combined.csv")
    out = out_dir or CFG.get("OUT_DIR", "./results")
    run_experiment(cfg=CFG, data_path=path, out_dir=out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OZON TA Pipeline")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    default_data = CFG.get("FILE_PATH", "OZON_combined.csv")
    if args.smoke:
        smoke_test(args.data or default_data)
    else:
        run(data_path=args.data, out_dir=args.out)
