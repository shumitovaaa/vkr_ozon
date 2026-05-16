"""
Сводные таблицы для честного A/B сравнения «без новостей» vs «с новостями».

Логика модуля изолирована от обучения: на вход подаются уже посчитанные результаты
``run_ml`` и ``forecast_df`` для ARIMA-GARCH из двух прогонов с одним и тем же
walk-forward (одинаковые сплиты, гиперпараметры, seed) — отличается только наличие
новостных признаков в матрице ``X``. Разница метрик (Δ) считается как
``with_news − without_news``: для метрик-минимизации (MAE, RMSE, MAPE) отрицательная
Δ означает улучшение, для метрик-максимизации (R2, Accuracy, F1, AUC, MDA_%, IC) —
улучшение даёт положительная Δ.

Функции возвращают ``pandas.DataFrame``, готовые к сохранению в CSV.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from evaluation.metrics import regression_metrics
from evaluation.wf_selection import mean_metrics_by_model
from models.walk_forward import (
    ARIMA_GARCH_MODEL_LABEL,
    COL_MU_FORECAST,
    COL_Y_REALIZED,
    arima_garch_classification_from_forecasts,
)

logger = logging.getLogger(__name__)

# Метрики, по которым меньше = лучше (для интерпретации Δ).
_MINIMIZE_METRICS: frozenset[str] = frozenset({"MAE", "RMSE", "MAPE"})

# Метрики, которые сохраняются в сводных таблицах.
REGRESSION_METRIC_COLS: tuple[str, ...] = (
    "MAE",
    "RMSE",
    "MAPE",
    "MDA_%",
    "R2",
    "IC",
)
CLASSIFICATION_METRIC_COLS: tuple[str, ...] = (
    "Accuracy",
    "F1",
    "AUC",
)


def _delta_direction(metric: str) -> str:
    """Подсказка интерпретации Δ для строки из сводной таблицы."""
    return "lower_is_better" if metric in _MINIMIZE_METRICS else "higher_is_better"


def _improved_flag(metric: str, delta: float) -> Optional[bool]:
    """True, если ``delta`` соответствует улучшению при включении новостей."""
    if delta is None or (isinstance(delta, float) and np.isnan(delta)):
        return None
    if metric in _MINIMIZE_METRICS:
        return bool(delta < 0)
    return bool(delta > 0)


def _ml_metric_table_per_horizon(
    ml: Dict[str, Any],
    *,
    task: str,
    horizons: Sequence[int],
) -> pd.DataFrame:
    """Аккуратно достать средние WF-метрики «модель×горизонт» из результата run_ml."""
    if task not in ("regression", "classification"):
        raise ValueError(f"task должен быть 'regression' или 'classification', получено {task!r}")

    raw_key = "wf_reg" if task == "regression" else "wf_clf"
    rows: List[pd.DataFrame] = []
    raw_dict = ml.get(raw_key, {}) or {}

    for h in horizons:
        df = raw_dict.get(int(h))
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        agg = mean_metrics_by_model(df)
        if agg.empty:
            continue
        agg = agg.copy()
        agg["horizon"] = int(h)
        rows.append(agg)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out[[c for c in ["horizon", "model", *out.columns] if c in out.columns]]
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def _join_no_with(
    df_no: pd.DataFrame,
    df_with: pd.DataFrame,
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    """Слить две WF-таблицы по (horizon, model) и вычислить Δ по списку метрик."""
    if df_no.empty and df_with.empty:
        return pd.DataFrame()

    keep_no = ["horizon", "model", *[m for m in metric_cols if m in df_no.columns]]
    keep_with = ["horizon", "model", *[m for m in metric_cols if m in df_with.columns]]

    a = df_no[keep_no].rename(
        columns={m: f"{m}_no_news" for m in metric_cols if m in df_no.columns}
    )
    b = df_with[keep_with].rename(
        columns={m: f"{m}_with_news" for m in metric_cols if m in df_with.columns}
    )

    merged = a.merge(b, on=["horizon", "model"], how="outer")

    long_rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        h = int(row["horizon"])
        m_name = str(row["model"])
        for metric in metric_cols:
            no_v = row.get(f"{metric}_no_news")
            with_v = row.get(f"{metric}_with_news")
            no_f = float(no_v) if pd.notna(no_v) else float("nan")
            with_f = float(with_v) if pd.notna(with_v) else float("nan")
            delta = (
                with_f - no_f
                if not (np.isnan(no_f) or np.isnan(with_f))
                else float("nan")
            )
            improved = _improved_flag(metric, delta)
            long_rows.append(
                {
                    "horizon": h,
                    "model": m_name,
                    "metric": metric,
                    "no_news": no_f,
                    "with_news": with_f,
                    "delta_with_minus_no": delta,
                    "direction": _delta_direction(metric),
                    "improved_by_news": improved,
                }
            )
    return pd.DataFrame(long_rows).sort_values(["horizon", "model", "metric"]).reset_index(drop=True)


def build_ml_regression_comparison(
    ml_no_news: Dict[str, Any],
    ml_with_news: Dict[str, Any],
    horizons: Sequence[int],
) -> pd.DataFrame:
    """
    Длинная таблица сравнения регрессии «без новостей» vs «с новостями» для каждой
    модели на каждом горизонте по всем метрикам ``REGRESSION_METRIC_COLS``.
    """
    df_no = _ml_metric_table_per_horizon(ml_no_news, task="regression", horizons=horizons)
    df_with = _ml_metric_table_per_horizon(ml_with_news, task="regression", horizons=horizons)
    return _join_no_with(df_no, df_with, REGRESSION_METRIC_COLS)


def build_ml_classification_comparison(
    ml_no_news: Dict[str, Any],
    ml_with_news: Dict[str, Any],
    horizons: Sequence[int],
) -> pd.DataFrame:
    """
    Длинная таблица сравнения классификации (RF/LGB) «без новостей» vs «с новостями».
    """
    df_no = _ml_metric_table_per_horizon(ml_no_news, task="classification", horizons=horizons)
    df_with = _ml_metric_table_per_horizon(ml_with_news, task="classification", horizons=horizons)
    return _join_no_with(df_no, df_with, CLASSIFICATION_METRIC_COLS)


def build_arima_regression_table(
    forecast_df: pd.DataFrame,
    horizons: Sequence[int],
    *,
    label_prefix: str = ARIMA_GARCH_MODEL_LABEL,
) -> pd.DataFrame:
    """
    Регрессионные метрики ARIMA(-GARCH) для каждого горизонта (одна модель, без A/B,
    т.к. ARIMA-GARCH работает на ряде ``log_ret`` и не использует новостные признаки).
    """
    if forecast_df is None or forecast_df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for h in horizons:
        if int(h) == 1:
            cy, cm = COL_Y_REALIZED, COL_MU_FORECAST
        else:
            cy, cm = f"y_realized_h{int(h)}", f"mu_forecast_h{int(h)}"
        if cy not in forecast_df.columns or cm not in forecast_df.columns:
            continue
        sub = forecast_df[[cy, cm]].dropna()
        if len(sub) < 2:
            continue
        m = regression_metrics(
            y_true=sub[cy].to_numpy(dtype=np.float64),
            y_pred=sub[cm].to_numpy(dtype=np.float64),
            label=f"{label_prefix}_h{int(h)}",
        )
        m["horizon"] = int(h)
        m["n_obs"] = int(len(sub))
        rows.append(m)

    return pd.DataFrame(rows)


def build_arima_classification_table(
    forecast_df: pd.DataFrame,
    horizons: Sequence[int],
    *,
    label_prefix: str = ARIMA_GARCH_MODEL_LABEL,
) -> pd.DataFrame:
    """Тонкая обёртка над :func:`arima_garch_classification_from_forecasts` → DataFrame."""
    rows = arima_garch_classification_from_forecasts(
        forecast_df, horizons, label_prefix=label_prefix
    )
    return pd.DataFrame(rows)


def winners_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Свернуть длинную таблицу A/B по (horizon, metric) к строкам с долей моделей,
    где новости улучшили метрику. Полезно для интерпретации «дают ли новости
    реальное улучшение».
    """
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame()
    df = comparison_df.dropna(subset=["delta_with_minus_no", "improved_by_news"]).copy()
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby(["horizon", "metric"], as_index=False)
        .agg(
            n_models=("model", "count"),
            n_improved_by_news=("improved_by_news", lambda s: int(np.sum(s.astype(bool)))),
            mean_delta=("delta_with_minus_no", "mean"),
            median_delta=("delta_with_minus_no", "median"),
        )
        .sort_values(["horizon", "metric"])
        .reset_index(drop=True)
    )
    grouped["share_improved_by_news"] = grouped["n_improved_by_news"] / grouped["n_models"].replace(0, np.nan)
    return grouped


def assemble_comparison_summary(
    *,
    ml_reg: pd.DataFrame,
    ml_clf: pd.DataFrame,
    arima_reg_no_news: pd.DataFrame,
    arima_reg_with_news: pd.DataFrame,
    arima_clf_no_news: pd.DataFrame,
    arima_clf_with_news: pd.DataFrame,
) -> pd.DataFrame:
    """
    Объединить все длинные таблицы в один CSV для отчёта (одна строка =
    «горизонт × модель × метрика»). ARIMA-GARCH добавляется отдельно: для него
    значения ``no_news`` и ``with_news`` совпадают (новости не входят в ряд),
    Δ всегда 0 — это нужно для прозрачности отчёта.
    """
    parts: List[pd.DataFrame] = []
    if ml_reg is not None and not ml_reg.empty:
        tmp = ml_reg.copy()
        tmp.insert(0, "task", "regression")
        tmp.insert(0, "family", "ML")
        parts.append(tmp)
    if ml_clf is not None and not ml_clf.empty:
        tmp = ml_clf.copy()
        tmp.insert(0, "task", "classification")
        tmp.insert(0, "family", "ML")
        parts.append(tmp)

    def _wide_arima(
        df_no: pd.DataFrame,
        df_with: pd.DataFrame,
        metric_cols: Sequence[str],
        task: str,
    ) -> pd.DataFrame:
        if (df_no is None or df_no.empty) and (df_with is None or df_with.empty):
            return pd.DataFrame()
        a = df_no[["horizon", "model", *[c for c in metric_cols if c in df_no.columns]]].copy() if df_no is not None else pd.DataFrame()
        b = df_with[["horizon", "model", *[c for c in metric_cols if c in df_with.columns]]].copy() if df_with is not None else pd.DataFrame()
        # Проверяем модель-метку для совмещения; ARIMA одна и та же.
        long_rows: List[Dict[str, Any]] = []
        models = (
            sorted(set(a["model"].astype(str)) | set(b["model"].astype(str)))
            if not (a.empty and b.empty)
            else []
        )
        for model in models:
            sub_a = a[a["model"] == model] if not a.empty else pd.DataFrame()
            sub_b = b[b["model"] == model] if not b.empty else pd.DataFrame()
            horizons = sorted(set(sub_a.get("horizon", pd.Series(dtype=int)).tolist())
                              | set(sub_b.get("horizon", pd.Series(dtype=int)).tolist()))
            for h in horizons:
                row_a = sub_a[sub_a["horizon"] == h]
                row_b = sub_b[sub_b["horizon"] == h]
                for metric in metric_cols:
                    no_v = float(row_a[metric].iloc[0]) if not row_a.empty and metric in row_a.columns and pd.notna(row_a[metric].iloc[0]) else float("nan")
                    with_v = float(row_b[metric].iloc[0]) if not row_b.empty and metric in row_b.columns and pd.notna(row_b[metric].iloc[0]) else float("nan")
                    delta = (with_v - no_v) if not (np.isnan(no_v) or np.isnan(with_v)) else float("nan")
                    long_rows.append(
                        {
                            "family": "ARIMA-GARCH",
                            "task": task,
                            "horizon": int(h),
                            "model": str(model),
                            "metric": metric,
                            "no_news": no_v,
                            "with_news": with_v,
                            "delta_with_minus_no": delta,
                            "direction": _delta_direction(metric),
                            "improved_by_news": _improved_flag(metric, delta),
                        }
                    )
        return pd.DataFrame(long_rows)

    arima_reg_long = _wide_arima(
        arima_reg_no_news, arima_reg_with_news, REGRESSION_METRIC_COLS, "regression"
    )
    arima_clf_long = _wide_arima(
        arima_clf_no_news, arima_clf_with_news, CLASSIFICATION_METRIC_COLS, "classification"
    )
    if not arima_reg_long.empty:
        parts.append(arima_reg_long)
    if not arima_clf_long.empty:
        parts.append(arima_clf_long)

    if not parts:
        return pd.DataFrame()
    full = pd.concat(parts, ignore_index=True)
    cols_order = [
        "family",
        "task",
        "horizon",
        "model",
        "metric",
        "no_news",
        "with_news",
        "delta_with_minus_no",
        "direction",
        "improved_by_news",
    ]
    cols = [c for c in cols_order if c in full.columns] + [c for c in full.columns if c not in cols_order]
    return full[cols].sort_values(["family", "task", "horizon", "model", "metric"]).reset_index(drop=True)


def safe_to_csv(df: pd.DataFrame, path: Any, *, log_tag: str = "[AB]") -> bool:
    """Записать df в CSV; вернуть True при успехе, иначе False (с логированием)."""
    try:
        df.to_csv(path, index=False, encoding="utf-8")
        logger.info("%s Сохранено: %s", log_tag, path)
        return True
    except Exception as exc:
        logger.warning("%s Не удалось сохранить %s: %s", log_tag, path, exc)
        return False


__all__ = [
    "REGRESSION_METRIC_COLS",
    "CLASSIFICATION_METRIC_COLS",
    "build_ml_regression_comparison",
    "build_ml_classification_comparison",
    "build_arima_regression_table",
    "build_arima_classification_table",
    "winners_table",
    "assemble_comparison_summary",
    "safe_to_csv",
]
