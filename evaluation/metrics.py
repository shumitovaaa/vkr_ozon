"""
Метрики регрессии, классификации и торговых стратегий для лог-доходностей.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


def mda(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Доля совпадений знака факта и прогноза, в процентах."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)


def information_coefficient(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Spearman IC и p-value (типичная метрика для рангового сигнала)."""
    ic, pval = spearmanr(y_true, y_pred)
    return {
        "IC": round(float(ic), 4),
        "IC_pvalue": round(float(pval), 4),
    }


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
) -> Dict[str, Any]:
    """
    MAE, RMSE, MAPE (%), MDA%, R², Spearman IC для регрессии доходности.

    MAPE: mean |y - ŷ| / (|y| + eps) · 100; при очень малых |y| может раздуваться.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    mda_val = mda(y_true, y_pred)
    ic_dict = information_coefficient(y_true, y_pred)
    eps = 1e-8
    mape = float(
        np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0
    )

    return {
        "model": label,
        "MAE": round(mae, 6),
        "RMSE": round(rmse, 6),
        "MAPE": round(mape, 4),
        "MDA_%": round(mda_val, 2),
        "R2": round(r2, 4),
        "IC": ic_dict["IC"],
        "IC_pval": ic_dict["IC_pvalue"],
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label: str,
) -> Dict[str, Any]:
    """Accuracy, F1, AUC для направления движения."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return {
        "model": label,
        "Accuracy": round(float(acc), 4),
        "F1": round(float(f1), 4),
        "AUC": round(float(auc), 4),
    }


def trading_metrics(
    y_true_ret: np.ndarray,
    y_pred_direction: np.ndarray,
    label: str,
    commission: float = 0.0005,
) -> Dict[str, Any]:
    """
    Доходность стратегии long/short по знаку прогноза с комиссией на смену позиции.

    Sharpe/Sortino/Calmar в пересчёте на год с sqrt(252).
    """
    pos_changes = np.abs(np.diff(np.concatenate([[0], y_pred_direction])))
    trade_costs = pos_changes * commission
    strat_ret = y_true_ret * y_pred_direction - trade_costs

    cum = np.exp(np.cumsum(strat_ret))
    n = max(len(strat_ret), 1)
    annual_ret = float((cum[-1] ** (252 / n) - 1) * 100)
    running_max = np.maximum.accumulate(cum)
    max_dd = float(((cum - running_max) / (running_max + 1e-8)).min() * 100)

    mean_r, std_r = strat_ret.mean(), strat_ret.std()
    sharpe = float(mean_r / (std_r + 1e-8) * np.sqrt(252))
    sortino_neg = strat_ret[strat_ret < 0].std()
    sortino = float(mean_r / (sortino_neg + 1e-8) * np.sqrt(252))
    calmar = float(annual_ret / abs(max_dd)) if max_dd != 0 else 0.0

    return {
        "model": label,
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "Calmar": round(calmar, 3),
        "AnnualRet_%": round(annual_ret, 2),
        "MaxDD_%": round(max_dd, 2),
    }


def buy_hold_metrics(y_true_ret: np.ndarray) -> Dict[str, Any]:
    """Бенчмарк купи-и-держи по той же траектории доходностей."""
    cum = np.exp(np.cumsum(y_true_ret))
    n = max(len(y_true_ret), 1)
    annual_ret = float((cum[-1] ** (252 / n) - 1) * 100)
    running_max = np.maximum.accumulate(cum)
    max_dd = float(((cum - running_max) / (running_max + 1e-8)).min() * 100)
    mean_r = y_true_ret.mean()
    std_r = y_true_ret.std()
    sharpe = float(mean_r / (std_r + 1e-8) * np.sqrt(252))
    calmar = float(annual_ret / abs(max_dd)) if max_dd != 0 else 0.0
    return {
        "model": "Buy&Hold",
        "Sharpe": round(sharpe, 3),
        "Calmar": round(calmar, 3),
        "AnnualRet_%": round(annual_ret, 2),
        "MaxDD_%": round(max_dd, 2),
    }


def validate_residuals(residuals: np.ndarray, label: str) -> Dict[str, Any]:
    """
    Ljung-Box (белый шум) и ADF (стационарность остатков).

    Вызывается после walk-forward для диагностики недомоделированной структуры.
    """
    try:
        lb = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
        lb_p10 = float(lb["lb_pvalue"].iloc[0])
        lb_p20 = float(lb["lb_pvalue"].iloc[1])
    except Exception:
        lb_p10 = lb_p20 = float("nan")

    try:
        _, adf_pval, *_ = adfuller(residuals, autolag="AIC")
    except Exception:
        adf_pval = float("nan")

    is_white_noise = lb_p10 > 0.05
    is_stationary = adf_pval < 0.05

    if not is_white_noise:
        logger.warning(
            "[%s] Остатки автокоррелированы (Ljung-Box p=%.4f)",
            label,
            lb_p10,
        )
    if not is_stationary:
        logger.warning(
            "[%s] Остатки нестационарны (ADF p=%.4f)", label, adf_pval
        )

    return {
        "model": label,
        "LjungBox_p10": round(lb_p10, 4),
        "LjungBox_p20": round(lb_p20, 4),
        "ADF_pvalue": round(adf_pval, 4),
        "white_noise": is_white_noise,
        "stationary": is_stationary,
    }


class Evaluator:
    """Обёртка для обратной совместимости со статическими метриками."""

    regression_metrics = staticmethod(regression_metrics)
    classification_metrics = staticmethod(classification_metrics)
