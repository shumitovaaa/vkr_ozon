"""
Валидация и очистка временного ряда MOEX перед признаками.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Убрать дубликаты индекса, заполнить пропуски, залогировать выбросы.

    Выбросы по робастному z-score к лог-доходности не удаляются —
    экстремальные дни OZON (2022–2024) считаем реальными.

    Parameters
    ----------
    df
        OHLCV с DatetimeIndex.
    cfg
        Ожидается ключ OUTLIER_ZSCORE для порога логирования.

    Returns
    -------
    Очищенный копия DataFrame.
    """
    df = df.copy()

    n_dup = int(df.index.duplicated().sum())
    if n_dup:
        logger.warning("[VALIDATE] Удалено %s дубликатов индекса", n_dup)
        df = df[~df.index.duplicated(keep="last")]

    df = df.sort_index()

    bdays = pd.bdate_range(df.index.min(), df.index.max())
    missing_days = bdays.difference(df.index)
    if len(missing_days) > 0:
        logger.warning(
            "[VALIDATE] Пропущено %s рабочих дней (первые 5: %s)",
            len(missing_days),
            missing_days[:5].tolist(),
        )

    missing_report = df.isnull().sum()
    missing_report = missing_report[missing_report > 0]
    if not missing_report.empty:
        logger.warning(
            "[VALIDATE] Пропуски до заполнения:\n%s",
            missing_report.to_string(),
        )

    price_cols = [c for c in ["OPEN", "HIGH", "LOW", "CLOSE"] if c in df.columns]
    df[price_cols] = df[price_cols].ffill()
    if "VOL" in df.columns:
        med_vol = df["VOL"].rolling(20, min_periods=1).median()
        df["VOL"] = df["VOL"].fillna(med_vol)
    df = df.dropna(subset=price_cols)

    if "CLOSE" in df.columns:
        log_ret = np.log(df["CLOSE"] / df["CLOSE"].shift(1)).dropna()
        med = log_ret.median()
        mad = (log_ret - med).abs().median()
        z = (log_ret - med) / (mad * 1.4826 + 1e-8)
        threshold = cfg.get("OUTLIER_ZSCORE", 3.5)
        outliers = z[z.abs() > threshold]
        if not outliers.empty:
            logger.warning(
                "[VALIDATE] Выбросов (|z|>%s): %s", threshold, len(outliers)
            )
            logger.warning(
                "[VALIDATE] Даты: %s", outliers.index.tolist()[:10]
            )

    logger.info(
        "[VALIDATE] Итог: %s строк, %s колонок, период %s–%s",
        len(df),
        df.shape[1],
        df.index.min().date(),
        df.index.max().date(),
    )
    return df
