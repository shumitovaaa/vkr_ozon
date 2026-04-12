"""
Единая формула лог-доходности по ряду цен закрытия.
"""

from __future__ import annotations

import numpy as np
from pandas import Series


def compute_log_returns(close: Series) -> Series:
    """
    ln(C_t / C_{t-1}); первая строка — NaN (как у ``shift(1)``).

    Parameters
    ----------
    close
        Ряд цен закрытия (индекс — даты).
    """
    return np.log(close.astype(float) / close.astype(float).shift(1))
