"""
Технические индикаторы и календарные признаки для OZON.

Сезонность: дневные доходности слабо периодичны; признаки month/dow —
мягкий учёт календарных эффектов (в т.ч. эмпирические дни/месяцы из EDA).

Выбросы на входе не режем — они уже отфильтрованы/залогированы в validator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


def compute_indicators(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Добавить к df индикаторы (ta), волатильность, HAR-RV, OBV_z, календарь.

    Parameters
    ----------
    df
        OHLCV с индексом-датами.
    cfg
        Периоды SMA, RSI, ATR, OBV_WINDOW и т.д.

    Returns
    -------
    Расширенный DataFrame.
    """
    out = df.copy()

    close = df["CLOSE"]
    high = df["HIGH"]
    low = df["LOW"]
    vol = df["VOL"]
    log_ret = np.log(close / close.shift(1))

    n_lags = max(0, int(cfg.get("N_LAGS", 0)))
    for k in range(1, n_lags + 1):
        out[f"log_ret_lag{k}"] = log_ret.shift(int(k))

    for p in cfg.get("SMA_PERIODS", [20, 50, 200]):
        out[f"sma{p}"] = ta.trend.sma_indicator(close, window=int(p))

    out["price_vs_sma20"] = (close - out["sma20"]) / (out["sma20"] + 1e-8)

    out["RSI14"] = ta.momentum.rsi(close, window=int(cfg.get("RSI_PERIOD", 14)))
    stoch = ta.momentum.StochasticOscillator(
        high, low, close, window=14, smooth_window=3
    )
    out["STOCHK"] = stoch.stoch()
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    out["BBpctb"] = bb.bollinger_pband()
    out["BBwidth"] = bb.bollinger_wband()

    macd_obj = ta.trend.MACD(close)
    out["MACDnorm"] = macd_obj.macd_diff() / (close + 1e-8)

    atr = ta.volatility.AverageTrueRange(
        high,
        low,
        close,
        window=int(cfg.get("ATR_PERIOD", 14)),
    )
    out["ATRnorm"] = atr.average_true_range() / (close + 1e-8)

    out["vol_std5"] = log_ret.rolling(5).std()
    out["vol_std20"] = log_ret.rolling(20).std()

    rv_daily = log_ret**2
    out["har_d"] = rv_daily.shift(1)
    out["har_w"] = rv_daily.shift(1).rolling(5).mean()
    out["har_m"] = rv_daily.shift(1).rolling(22).mean()

    out["rv_park5"] = (
        (np.log(high / low) ** 2) / (4 * np.log(2))
    ).rolling(5).mean()

    out["ret_sq_lag1"] = log_ret.shift(1) ** 2
    out["sign_lag1"] = np.sign(log_ret.shift(1))
    out["ret_abs_ma5"] = log_ret.abs().rolling(5).mean()
    out["ret_abs_ma20"] = log_ret.abs().rolling(20).mean()

    obv = ta.volume.on_balance_volume(close, vol)
    obv_norm_w = int(cfg.get("OBV_NORM_WINDOW", cfg.get("OBV_WINDOW", 20)))
    obv_norm_w = max(2, obv_norm_w)
    obv_ma = obv.rolling(obv_norm_w).mean()
    obv_std = obv.rolling(obv_norm_w).std()
    out["OBV_z"] = (obv - obv_ma) / (obv_std + 1e-8)

    idx = df.index
    out["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 5)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 5)
    out["is_month_end"] = idx.is_month_end.astype(int)
    out["is_quarter_end"] = idx.is_quarter_end.astype(int)
    out["is_friday"] = (idx.dayofweek == 4).astype(int)
    out["is_march"] = (idx.month == 3).astype(int)
    out["is_july"] = (idx.month == 7).astype(int)
    out["is_oct"] = (idx.month == 10).astype(int)
    out["in_crisis"] = ((idx >= "2022-02-24") & (idx <= "2022-12-31")).astype(
        int
    )

    logger.info("[INDICATORS] Признаки вычислены: %s колонок", out.shape[1])
    return out
