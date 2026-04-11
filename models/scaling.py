"""
Общее масштабирование признаков для walk-forward и стекинга.

RobustScaler обучается только на train — исключает утечку статистик теста.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.preprocessing import RobustScaler


def robust_scale_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    *,
    quantile_range: Tuple[int, int] = (10, 90),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обучить RobustScaler на X_train и преобразовать train и test.

    Parameters
    ----------
    X_train, X_test
        Матрицы признаков с одинаковыми именами колонок.
    quantile_range
        Квантили для устойчивости к выбросам (как в sklearn).

    Returns
    -------
    (Xs_train, Xs_test)
        Преобразованные DataFrame с сохранённым индексом и колонками.
    """
    scaler = RobustScaler(quantile_range=quantile_range).fit(X_train)
    Xs_tr = pd.DataFrame(
        scaler.transform(X_train),
        index=X_train.index,
        columns=X_train.columns,
    )
    Xs_te = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns,
    )
    return Xs_tr, Xs_te
