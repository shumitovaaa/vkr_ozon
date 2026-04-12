"""
Агрегация walk-forward метрик по регрессорам и выбор лучшей модели.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd


def mean_metrics_by_model(df_reg: pd.DataFrame) -> pd.DataFrame:
    """
    Средние метрики по фолдам для каждой модели (один горизонт).
    """
    if df_reg.empty or "model" not in df_reg.columns:
        return pd.DataFrame()
    num_cols = [
        c
        for c in df_reg.columns
        if c not in ("model", "fold", "horizon") and pd.api.types.is_numeric_dtype(df_reg[c])
    ]
    if not num_cols:
        return pd.DataFrame()
    g = df_reg.groupby("model", as_index=False)[num_cols].mean()
    return g


def select_best_model(
    agg: pd.DataFrame,
    metric: str = "RMSE",
) -> Tuple[str, Dict[str, Any]]:
    """
    Выбрать лучшую модель по средней метрике.

    Для R2, Accuracy, AUC, F1 — максимизация; для MAE, RMSE, MAPE — минимизация.
    """
    if agg.empty or "model" not in agg.columns or metric not in agg.columns:
        return "", {}

    _maximize = frozenset({"R2", "Accuracy", "AUC", "F1"})
    minimize = metric not in _maximize
    series = agg.set_index("model")[metric]
    if minimize:
        best_name = str(series.idxmin())
        best_val = float(series.min())
    else:
        best_name = str(series.idxmax())
        best_val = float(series.max())

    row = agg.loc[agg["model"] == best_name].iloc[0].to_dict()
    return best_name, {"metric": metric, "value": best_val, "row": row}


def overall_best_across_horizons(
    per_horizon_aggs: Dict[int, pd.DataFrame],
    metric: str = "RMSE",
) -> Tuple[str, Dict[str, Any]]:
    """
    Одна лучшая модель в среднем по всем горизонтам: среднее metric по h, затем argmin/max.

    На входе — таблицы ``mean_metrics_by_model`` по каждому h (уже усреднение по фолдам).
    """
    rows: List[pd.DataFrame] = []
    for _, df in per_horizon_aggs.items():
        if df is None or df.empty:
            continue
        rows.append(df)
    if not rows:
        return "", {}
    full = pd.concat(rows, ignore_index=True)
    g = full.groupby("model", as_index=False)[metric].mean()
    return select_best_model(g, metric=metric)
