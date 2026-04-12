"""
Загрузка CSV-котировок MOEX (OHLCV).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Set, Union

import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLS = frozenset({"OPEN", "HIGH", "LOW", "CLOSE", "VOL"})


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Загрузить CSV с котировками.

    Ожидаются колонки DATE, OPEN, HIGH, LOW, CLOSE, VOL (регистр не важен).
    Перебираются разделители ',', ';', '\\t'.

    Parameters
    ----------
    filepath
        Путь к файлу.

    Returns
    -------
    DataFrame с DatetimeIndex и числовыми OHLCV.

    Raises
    ------
    FileNotFoundError
        Файл отсутствует.
    ValueError
        Некорректный формат или обязательные колонки.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path.resolve()}")

    df: pd.DataFrame | None = None

    for sep in [",", ";", "\t"]:
        try:
            tmp = pd.read_csv(path, sep=sep, engine="python")
            if tmp.shape[1] >= 4:
                df = tmp
                logger.debug("Файл прочитан с разделителем %r", sep)
                break
        except Exception as exc:
            logger.debug("Попытка sep=%r не удалась: %s", sep, exc)

    if df is None:
        raise ValueError(f"Не удалось распарсить файл: {filepath}")

    df.columns = (
        df.columns.str.strip().str.upper().str.replace(r"[\s<>]", "", regex=True)
    )

    if "DATE" not in df.columns:
        raise ValueError(
            f"Колонка DATE не найдена. Доступные: {list(df.columns)}"
        )

    try:
        date_col = df["DATE"].astype(str).str.strip()
        # `infer_datetime_format` deprecated; поведение по умолчанию достаточно.
        df["DATE"] = pd.to_datetime(date_col)
    except Exception as exc:
        raise ValueError(f"Не удалось разобрать даты: {exc}") from exc

    df = df.sort_values("DATE").reset_index(drop=True)
    df.set_index("DATE", inplace=True)

    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_required = _REQUIRED_COLS - set(df.columns)
    if missing_required:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing_required}")

    df.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"], inplace=True)

    if len(df) == 0:
        raise ValueError("После удаления пропусков по OHLC не осталось ни одной строки")

    logger.info(
        "Загружено: %s строк, %s — %s",
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
    )
    return df


def load_trading_dates(filepath: Union[str, Path]) -> Set[date]:
    """
    Множество календарных дат из CSV котировок (после нормализации индекса к полуночи).

    Используется, чтобы оставлять только новости в дни, для которых есть цена в датасете.
    """
    df = load_data(filepath)
    return {pd.Timestamp(ts).normalize().date() for ts in df.index}
