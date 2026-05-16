"""
Общее масштабирование признаков для walk-forward и стекинга.

RobustScaler обучается только на train — исключает утечку статистик теста.
Опциональная винзоризация (``WINSORIZE_OUTLIERS``): обрезка по квантилям,
оценённым только на train, затем тот же clip для test — без утечки.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def _winsorize_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    low_pct: float = 5.0,
    high_pct: float = 95.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Обрезка по столбцам: границы с train (nanpercentile), применение к train и test."""
    X_tr = X_train.copy()
    X_te = X_test.copy()
    lo_q = float(min(low_pct, high_pct))
    hi_q = float(max(low_pct, high_pct))
    for col in X_tr.columns:
        arr = X_tr[col].to_numpy(dtype=np.float64, copy=False)
        lo = float(np.nanpercentile(arr, lo_q))
        hi = float(np.nanpercentile(arr, hi_q))
        if lo > hi:
            lo, hi = hi, lo
        X_tr[col] = X_tr[col].clip(lower=lo, upper=hi)
        X_te[col] = X_te[col].clip(lower=lo, upper=hi)
    return X_tr, X_te


def robust_scale_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    quantile_range: Tuple[int, int] = (10, 90),
    winsorize: bool = False,
    winsorize_low_pct: float = 5.0,
    winsorize_high_pct: float = 95.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обучить RobustScaler на X_train и преобразовать train и test.

    Parameters
    ----------
    X_train, X_test
        Матрицы признаков с одинаковыми именами колонок.
    quantile_range
        Квантили для устойчивости к выбросам (как в sklearn).
    winsorize
        Если True — перед scaler обрезка признаков по квантилям train
        (по умолчанию 5-й и 95-й процентили).
    winsorize_low_pct, winsorize_high_pct
        Процентили для винзоризации (только train для оценки границ).

    Returns
    -------
    (Xs_train, Xs_test)
        Преобразованные DataFrame с сохранённым индексом и колонками.
    """
    X_tr_in = X_train
    if winsorize:
        X_tr_in, X_test = _winsorize_train_test(
            X_train,
            X_test,
            low_pct=winsorize_low_pct,
            high_pct=winsorize_high_pct,
        )
    scaler = RobustScaler(quantile_range=quantile_range).fit(X_tr_in)
    Xs_tr = pd.DataFrame(
        scaler.transform(X_tr_in),
        index=X_tr_in.index,
        columns=X_tr_in.columns,
    )
    Xs_te = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns,
    )
    return Xs_tr, Xs_te
