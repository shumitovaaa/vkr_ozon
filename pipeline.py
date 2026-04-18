"""
Единая сборка пайплайна: EDA, baseline ARIMA–GARCH, walk-forward ML (Ridge, RF, LGB), стекинг.

Модели по смыслу: (1) эконометрика — ARIMA+GARCH по ряду доходностей; (2) ML-регрессия и
(3) ML-классификация направления — Ridge/RF/LGB в walk-forward; (4) опционально стекинг.
Все настройки — в ``config.CFG``.

Сравнение регрессоров — по единому walk-forward; лучшая модель на горизонт и в среднем по h.
Для ARIMA график строится по горизонту с лучшей метрикой среди ``ARIMA_FORECAST_HORIZONS``.

Точка входа: run_experiment().
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from statsmodels.tsa.stattools import adfuller

from data.loader import load_data
from data.validator import validate_data
from features.builder import FeatureBuilder
from features.indicators import compute_indicators
from news.merge import attach_news_sentiment_features
from evaluation.metrics import (
    buy_hold_metrics,
    regression_metrics,
    trading_metrics,
)
from evaluation.wf_selection import (
    mean_metrics_by_model,
    overall_best_across_horizons,
    select_best_model,
)
from features.returns import compute_log_returns
from models.walk_forward import (
    COL_CI_LOWER,
    COL_CI_UPPER,
    COL_MU_FORECAST,
    COL_Y_REALIZED,
    rolling_forecast_hybrid_arima_garch,
    walk_forward_train_eval,
)
from models.stacking import fit_stacking
from visualization.plots import Visualizer


def _arima_subframe_for_horizon(
    forecast_df: pd.DataFrame, h: int
) -> pd.DataFrame:
    """Вырезка прогноза ARIMA для горизонта h с колонками как у h=1."""
    h = int(h)
    if h == 1:
        return forecast_df[
            [COL_Y_REALIZED, COL_MU_FORECAST, COL_CI_LOWER, COL_CI_UPPER]
        ].copy()
    cy, cm, cl, ch = (
        f"y_realized_h{h}",
        f"mu_forecast_h{h}",
        f"ci_lower_h{h}",
        f"ci_upper_h{h}",
    )
    if cy not in forecast_df.columns:
        return pd.DataFrame()
    return forecast_df[[cy, cm, cl, ch]].rename(
        columns={
            cy: COL_Y_REALIZED,
            cm: COL_MU_FORECAST,
            cl: COL_CI_LOWER,
            ch: COL_CI_UPPER,
        }
    )


def _arima_horizon_metrics_df(
    forecast_df: pd.DataFrame, fh: Sequence[int]
) -> pd.DataFrame:
    """Сводка регрессионных метрик по тестовому окну для каждого h."""
    rows: list[dict[str, Any]] = []
    for h in fh:
        sub = _arima_subframe_for_horizon(forecast_df, int(h))
        sub = sub.dropna(subset=[COL_Y_REALIZED, COL_MU_FORECAST], how="any")
        if len(sub) < 2:
            continue
        r = regression_metrics(
            sub[COL_Y_REALIZED].values,
            sub[COL_MU_FORECAST].values,
            f"ARIMA_h{h}",
        )
        rows.append(
            {
                "horizon": int(h),
                "model": f"ARIMA_h{h}",
                "MAE": r.get("MAE"),
                "RMSE": r.get("RMSE"),
                "MAPE": r.get("MAPE"),
                "R2": r.get("R2"),
            }
        )
    return pd.DataFrame(rows)


logger = logging.getLogger(__name__)


def export_eda_stats(
    df_feat: pd.DataFrame, out: Path, cfg: Dict[str, Any]
) -> None:
    """
    Сохранить численные EDA-артефакты: ADF, сезонность, корреляции, новости.
    """
    _ = cfg
    out.mkdir(parents=True, exist_ok=True)
    if "CLOSE" not in df_feat.columns:
        logger.warning("[EDA] Нет CLOSE — пропуск export_eda_stats")
        return

    close = df_feat["CLOSE"].astype(float)
    log_ret = (
        df_feat["log_ret"].astype(float)
        if "log_ret" in df_feat.columns
        else compute_log_returns(close)
    )
    log_ret = log_ret.dropna()
    if log_ret.empty:
        logger.warning("[EDA] Пустой ряд log_ret — пропуск export_eda_stats")
        return

    adf_summary: Dict[str, Any] = {}
    try:
        adf_stat, adf_p, adf_lag, adf_nobs, adf_crit, *_ = adfuller(
            log_ret, autolag="AIC"
        )
        adf_summary = {
            "adf_statistic": float(adf_stat),
            "p_value": float(adf_p),
            "used_lag": int(adf_lag),
            "nobs": int(adf_nobs),
            "critical_values": {k: float(v) for k, v in adf_crit.items()},
            "log_ret_mean": float(log_ret.mean()),
            "log_ret_std": float(log_ret.std()),
            "log_ret_skew": float(skew(log_ret, bias=False)),
            "log_ret_kurtosis": float(kurtosis(log_ret, fisher=True, bias=False)),
            "n_obs": int(len(log_ret)),
        }
    except Exception as exc:
        logger.warning("[EDA] ADF calc failed: %s", exc)
        adf_summary = {"error": str(exc), "n_obs": int(len(log_ret))}

    try:
        (out / "eda_adf_log_ret.json").write_text(
            json.dumps(adf_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("[EDA] save eda_adf_log_ret.json: %s", exc)

    try:
        idx = df_feat.index if isinstance(df_feat.index, pd.DatetimeIndex) else None
        if idx is None:
            tmp_dates = pd.to_datetime(df_feat.get("date"), errors="coerce")
            idx = pd.DatetimeIndex(tmp_dates)
        base = pd.DataFrame({"log_ret": log_ret})
        base["weekday"] = base.index.dayofweek
        base["month"] = base.index.month
        weekday = (
            base.groupby("weekday")["log_ret"].mean().mul(100.0).reset_index()
        )
        weekday["weekday_name"] = weekday["weekday"].map(
            {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
        )
        month = base.groupby("month")["log_ret"].mean().mul(100.0).reset_index()
        month["month_name"] = month["month"].map(
            {
                1: "Jan",
                2: "Feb",
                3: "Mar",
                4: "Apr",
                5: "May",
                6: "Jun",
                7: "Jul",
                8: "Aug",
                9: "Sep",
                10: "Oct",
                11: "Nov",
                12: "Dec",
            }
        )
        weekday.rename(columns={"log_ret": "mean_log_ret_pct"}, inplace=True)
        month.rename(columns={"log_ret": "mean_log_ret_pct"}, inplace=True)
        weekday.to_csv(out / "eda_seasonality_weekday.csv", index=False, encoding="utf-8")
        month.to_csv(out / "eda_seasonality_month.csv", index=False, encoding="utf-8")
    except Exception as exc:
        logger.warning("[EDA] seasonality export failed: %s", exc)

    try:
        numeric = df_feat.select_dtypes(include=[np.number]).copy()
        if not numeric.empty:
            numeric.corr(method="pearson").to_csv(
                out / "eda_corr_pearson.csv", encoding="utf-8"
            )
            numeric.corr(method="spearman").to_csv(
                out / "eda_corr_spearman.csv", encoding="utf-8"
            )
    except Exception as exc:
        logger.warning("[EDA] corr export failed: %s", exc)

    try:
        if "sentiment_score" in df_feat.columns:
            s = df_feat["sentiment_score"].dropna().astype(float)
            if not s.empty:
                cls = pd.Series(np.where(s > 0.05, "positive", np.where(s < -0.05, "negative", "neutral")))
                cls_counts = cls.value_counts().rename_axis("sentiment_class").reset_index(name="count")
                cls_counts.to_csv(
                    out / "news_sentiment_class_counts.csv",
                    index=False,
                    encoding="utf-8",
                )
    except Exception as exc:
        logger.warning("[EDA] news sentiment counts export failed: %s", exc)


def run_eda(df_feat: pd.DataFrame, out: Path, cfg: Dict[str, Any]) -> None:
    """
    Построить EDA-графики в каталог out.

    Parameters
    ----------
    df_feat
        Данные с индикаторами.
    out
        Каталог для PNG.
    cfg
        Конфигурация (зарезервировано для будущих флагов EDA).
    """
    _ = cfg
    viz = Visualizer(out)
    viz.plot_price_volume(df_feat)
    viz.plot_returns(df_feat)
    viz.plot_seasonality(df_feat)
    viz.plot_correlation(df_feat)
    try:
        viz.plot_news_sentiment(df_feat)
    except Exception as exc:
        logger.warning("[EDA] plot_news_sentiment: %s", exc)
    if "CLOSE" in df_feat.columns:
        viz.plot_acf_pacf(df_feat["CLOSE"], "close")
    if "log_ret" in df_feat.columns:
        viz.plot_acf_pacf(df_feat["log_ret"].dropna(), "log_ret")


def run_hybrid_arima_garch(
    df: pd.DataFrame, cfg: Dict[str, Any], out: Path
) -> pd.DataFrame:
    """
    Rolling baseline: для h=1 — гибрид ARIMA+GARCH; для h∈ARIMA_FORECAST_HORIZONS, h>1 —
    кумулятивный прогноз ARIMA (как ``reg_h``).

    По метрике ``ARIMA_BEST_HORIZON_METRIC`` (или ``BEST_MODEL_METRIC``) выбирается
    лучший h среди заданных; сохраняется один график ``fig_forecast_arima_garch_best.png``
    и JSON ``arima_best_horizon.json`` с метриками по всем h.

    Returns
    -------
    DataFrame: колонки h=1 — ``y_realized``, ``mu_forecast``, …; для h>1 —
    ``y_realized_h{h}``, ``mu_forecast_h{h}``, ``ci_lower_h{h}``, ``ci_upper_h{h}``.
    """
    if "log_ret" not in df.columns:
        df = df.copy()
        df["log_ret"] = compute_log_returns(df["CLOSE"])

    series = df["log_ret"].dropna()
    n_sr = len(series)
    if n_sr < 2:
        logger.warning(
            "[ARIMA-GARCH] слишком мало точек лог-доходности (%s) — пропуск rolling",
            n_sr,
        )
        return pd.DataFrame()

    min_train = int(cfg.get("ARIMA_GARCH_MIN_TRAIN", 100))
    test_frac = float(cfg.get("TEST_FRACTION", 0.2))
    test_size = max(1, int(n_sr * test_frac))
    if cfg.get("ARIMA_MAX_TEST_DAYS"):
        test_size = min(test_size, int(cfg["ARIMA_MAX_TEST_DAYS"]))
    test_size = max(30, test_size)
    cap_len = max(1, n_sr - 1)
    if n_sr > min_train:
        test_size = min(test_size, cap_len, n_sr - min_train)
    else:
        test_size = min(test_size, cap_len)
    if test_size < 1:
        test_size = 1
    logger.info(
        "[ARIMA-GARCH] test_size=%s из len(log_ret)=%s (%.1f%% теста)",
        test_size,
        n_sr,
        100.0 * test_size / max(n_sr, 1),
    )

    order_cfg = cfg.get("ARIMA_ORDER", [1, 0, 1])
    arima_order = tuple(int(x) for x in order_cfg[:3])
    if len(arima_order) < 3:
        arima_order = (1, 0, 1)

    fh = list(cfg.get("ARIMA_FORECAST_HORIZONS", [1, 5, 10]))
    forecast_df = rolling_forecast_hybrid_arima_garch(
        y=series,
        test_size=test_size,
        arima_order=arima_order,
        alpha=float(cfg.get("CI_ALPHA", 0.10)),
        garch_p=int(cfg.get("GARCH_P", 1)),
        garch_q=int(cfg.get("GARCH_Q", 1)),
        min_train_for_hybrid=int(cfg.get("ARIMA_GARCH_MIN_TRAIN", 100)),
        use_garch_on_eps=bool(cfg.get("USE_ARIMA_GARCH", True)),
        forecast_horizons=fh,
    )

    metric_key = cfg.get("ARIMA_BEST_HORIZON_METRIC") or cfg.get(
        "BEST_MODEL_METRIC", "RMSE"
    )
    metric_key = str(metric_key)
    if metric_key not in ("MAE", "RMSE", "MAPE", "R2"):
        logger.warning(
            "[ARIMA-GARCH] метрика %s для выбора h не поддерживается, используем RMSE",
            metric_key,
        )
        metric_key = "RMSE"
    mdf = _arima_horizon_metrics_df(forecast_df, fh)
    best_h = int(fh[0]) if fh else 1
    best_meta: Dict[str, Any] = {}
    if not mdf.empty and metric_key in mdf.columns:
        _name, best_meta = select_best_model(mdf, metric=metric_key)
        hm = re.match(r"ARIMA_h(\d+)", str(_name))
        if hm:
            best_h = int(hm.group(1))
    else:
        if mdf.empty:
            logger.warning(
                "[ARIMA-GARCH] не удалось посчитать метрики по горизонтам — график h=%s",
                best_h,
            )

    try:
        (out / "arima_horizons_metrics.csv").write_text(
            mdf.to_csv(index=False, encoding="utf-8"),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("[ARIMA-GARCH] сохранение arima_horizons_metrics.csv: %s", exc)

    try:
        summary = {
            "selection_metric": metric_key,
            "horizons_evaluated": [int(x) for x in fh],
            "metrics_by_horizon": mdf.to_dict("records") if not mdf.empty else [],
            "best_horizon": best_h,
            "best_metric_value": best_meta.get("value"),
        }
        (out / "arima_best_horizon.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("[ARIMA-GARCH] сохранение arima_best_horizon.json: %s", exc)

    viz = Visualizer(out)
    label = cfg.get("ARIMA_PLOT_LABEL", "ARIMA-GARCH")
    sub_best = _arima_subframe_for_horizon(forecast_df, best_h)
    sub_best = sub_best.dropna(subset=[COL_Y_REALIZED, COL_MU_FORECAST], how="any")
    extra_line = (
        f"Лучший горизонт по {metric_key} среди {{{', '.join(str(x) for x in fh)}}} "
        f"(значение={best_meta.get('value')})"
        if best_meta
        else f"Горизонт h={best_h} (метрики по горизонтам недоступны)"
    )
    try:
        if not sub_best.empty:
            viz.plot_arima_garch_forecast(
                y_full=series,
                forecast_df=sub_best,
                horizon=int(best_h),
                model_label=label,
                extra_title_line=extra_line,
                save_filename="fig_forecast_arima_garch_best.png",
            )
        else:
            logger.warning("[ARIMA-GARCH] пустая вырезка для лучшего h=%s", best_h)
    except Exception as exc:
        logger.warning("[ARIMA-GARCH] plot best h=%s failed: %s", best_h, exc)

    logger.info(
        "[ARIMA-GARCH] Лучший горизонт по %s: h=%s (%s)",
        metric_key,
        best_h,
        best_meta,
    )

    return forecast_df


def _prepare_targets_and_features(
    df_feat: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Собрать X и целевые ряды с общим индексом без NaN."""
    fb = FeatureBuilder(cfg)
    X = fb.build_features(df_feat)
    tg = fb.build_targets(df_feat)

    data = X.join(tg, how="inner").dropna()
    X_aligned = data[X.columns]
    tg_aligned = data[tg.columns]

    logger.info(
        "[PIPELINE] Features aligned: X=%s, targets=%s",
        X_aligned.shape,
        tg_aligned.shape,
    )
    return X_aligned, tg_aligned


def _wf_slug(model_name: str) -> str:
    """Безопасное имя файла для графиков (модель + горизонт задаются отдельно)."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", model_name.strip().lower()).strip("_")
    return s or "model"


def _plot_wf_best_only(
    viz: Visualizer,
    preds_df: pd.DataFrame,
    X: pd.DataFrame,
    y_reg: pd.Series,
    horizon: int,
    best_model: str,
) -> None:
    """Итоговые графики только для лучшей WF-модели (факт, прогноз, остатки)."""
    if not best_model:
        return
    pred_col = f"pred_{best_model}"
    if pred_col not in preds_df.columns:
        logger.warning("[ML] Нет колонки %s для лучшей модели h=%s", pred_col, horizon)
        return
    test_start = preds_df.index.min()
    train_mask = X.index < test_start
    train_dates = X.index[train_mask]
    train_true = y_reg.loc[train_mask].values
    slug = _wf_slug(best_model)
    try:
        viz.plot_wf_best_forecast(
            train_dates=train_dates,
            train_true=train_true,
            test_dates=preds_df.index,
            test_true=preds_df["y_true_reg"].values,
            test_pred=preds_df[pred_col].values,
            model_name=best_model,
            file_slug=slug,
            horizon=horizon,
        )
        viz.plot_wf_best_residuals_diagnostics(
            preds_df, best_model, slug, horizon
        )
    except Exception as exc:
        logger.warning("[ML] графики лучшей модели h=%s: %s", horizon, exc)


def _holdout_split(
    X: pd.DataFrame, y_reg: pd.Series, test_frac: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Последняя доля ряда — тест для стекинга."""
    n = len(X)
    if n < 2:
        return X.iloc[:0], X, y_reg.iloc[:0], y_reg
    split_point = max(1, min(int(n * (1 - test_frac)), n - 1))
    return (
        X.iloc[:split_point],
        X.iloc[split_point:],
        y_reg.iloc[:split_point],
        y_reg.iloc[split_point:],
    )


def _run_horizon_ml(
    h: int,
    X: pd.DataFrame,
    tg: pd.DataFrame,
    cfg: Dict[str, Any],
    viz: Visualizer,
    results: Dict[str, Any],
) -> None:
    """Walk-forward, бейзлайны, торговые метрики, графики и hold-out модели."""
    reg_col = f"reg_{h}"
    clf_col = f"clf_{h}"

    if reg_col not in tg.columns or clf_col not in tg.columns:
        logger.warning("[ML] Missing targets for horizon=%s", h)
        return

    y_reg = tg[reg_col]
    y_clf = tg[clf_col]

    logger.info("[ML] Walk-forward start: horizon=%s", h)

    res_reg, res_clf, preds_df = walk_forward_train_eval(
        X=X,
        y_reg=y_reg,
        y_clf=y_clf,
        cfg=cfg,
        horizon=h,
    )

    df_reg = pd.DataFrame(res_reg) if res_reg else pd.DataFrame()
    df_clf = pd.DataFrame(res_clf) if res_clf else pd.DataFrame()

    results["wf_reg"][h] = df_reg
    results["wf_clf"][h] = df_clf
    results["wf_preds"][h] = preds_df

    naive_pred = np.zeros(len(y_reg))
    results["baseline"][h] = regression_metrics(
        y_reg.values, naive_pred, "Naive(0)"
    )
    results["buy_hold"][h] = buy_hold_metrics(y_reg.values)

    if not df_clf.empty and "Accuracy" in df_clf.columns:
        agg_clf = mean_metrics_by_model(df_clf)
        best_name, _ = (
            select_best_model(agg_clf, metric="Accuracy")
            if not agg_clf.empty
            else ("", {})
        )
        pred_col = f"pred_{best_name}"
        if best_name and pred_col in preds_df.columns:
            results["trading"][h] = trading_metrics(
                y_true_ret=preds_df["y_true_reg"].values,
                y_pred_direction=np.where(
                    preds_df[pred_col].values > 0, 1.0, -1.0
                ),
                label=f"{best_name}_h{h}",
                commission=cfg.get("COMMISSION", 0.0005),
            )

    if not df_reg.empty:
        try:
            viz.plot_wf_metrics(df_reg, horizon=h)
        except Exception as exc:
            logger.warning("[ML] plot_wf_metrics h=%s failed: %s", h, exc)

    metric_key = str(cfg.get("BEST_MODEL_METRIC", "RMSE"))
    agg = mean_metrics_by_model(df_reg)
    results["wf_reg_agg"][h] = agg
    best_name, best_meta = select_best_model(agg, metric=metric_key)
    results["wf_best"][h] = {
        "model": best_name,
        "metric": best_meta.get("metric", metric_key),
        "metric_value": best_meta.get("value"),
        "mean_metrics_by_model": agg.to_dict("records") if not agg.empty else [],
    }
    if not agg.empty:
        try:
            agg_path = Path(viz.out) / f"wf_reg_mean_h{h}.csv"
            agg.to_csv(agg_path, index=False, encoding="utf-8")
            logger.info("[ML] Сохранены средние WF-метрики: %s", agg_path)
        except Exception as exc:
            logger.warning("[ML] сохранение wf_reg_mean h=%s: %s", h, exc)

    if not df_clf.empty:
        try:
            agg_clf = mean_metrics_by_model(df_clf)
            results["wf_clf_agg"][h] = agg_clf
            clf_path = Path(viz.out) / f"wf_clf_mean_h{h}.csv"
            agg_clf.to_csv(clf_path, index=False, encoding="utf-8")
            best_clf_name, best_clf_meta = (
                select_best_model(agg_clf, metric="Accuracy")
                if not agg_clf.empty
                else ("", {})
            )
            results["wf_best_clf"][h] = {
                "model": best_clf_name,
                "metric": "Accuracy",
                "metric_value": best_clf_meta.get("value"),
                "mean_metrics_by_model": (
                    agg_clf.to_dict("records") if not agg_clf.empty else []
                ),
            }
        except Exception as exc:
            logger.warning("[ML] сохранение wf_clf_mean h=%s: %s", h, exc)

    _plot_wf_best_only(viz, preds_df, X, y_reg, h, best_name)
    if best_name:
        logger.info(
            "[ML] Лучшая WF-модель h=%s по %s: %s (значение=%s)",
            h,
            metric_key,
            best_name,
            best_meta.get("value"),
        )

    test_frac = cfg.get("TEST_FRACTION", 0.2)
    X_tr, X_te, y_tr, y_te = _holdout_split(X, y_reg, test_frac)

    try:
        st = fit_stacking(
            X_train=X_tr,
            y_train=y_tr.values,
            X_test=X_te,
            y_test=y_te.values,
            cfg=cfg,
        )
        results["stacking"][h] = st
        pd.DataFrame([{"horizon": h, **st}]).to_csv(
            Path(viz.out) / f"stacking_h{h}.csv", index=False, encoding="utf-8"
        )
    except Exception as exc:
        logger.warning("[ML] fit_stacking failed for h=%s: %s", h, exc)


def run_ml(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    out: Path,
) -> Dict[str, Any]:
    """
    ML-блок: единый walk-forward по горизонтам (Ridge, RF, LGB), стекинг на hold-out, графики.

    Returns
    -------
    Словарь с ключами wf_reg, wf_clf, wf_preds, wf_reg_agg, wf_best, wf_best_overall,
    stacking, baseline, trading, buy_hold.
    """
    X, tg = _prepare_targets_and_features(df, cfg)

    results: Dict[str, Any] = {
        "wf_reg": {},
        "wf_clf": {},
        "wf_preds": {},
        "wf_reg_agg": {},
        "wf_clf_agg": {},
        "wf_best": {},
        "wf_best_clf": {},
        "wf_best_overall": {},
        "stacking": {},
        "baseline": {},
        "trading": {},
        "buy_hold": {},
    }

    viz = Visualizer(out)
    horizons = list(cfg.get("HORIZONS", [1, 5, 10]))
    for h in horizons:
        _run_horizon_ml(h, X, tg, cfg, viz, results)

    metric_key = str(cfg.get("BEST_MODEL_METRIC", "RMSE"))
    per_h = {
        h: results["wf_reg_agg"].get(h)
        for h in horizons
        if isinstance(results["wf_reg_agg"].get(h), pd.DataFrame)
        and not results["wf_reg_agg"].get(h).empty
    }
    ob_name, ob_meta = overall_best_across_horizons(per_h, metric=metric_key)
    results["wf_best_overall"] = {
        "model": ob_name,
        "metric": ob_meta.get("metric", metric_key),
        "metric_value": ob_meta.get("value"),
    }
    if ob_name:
        logger.info(
            "[ML] Лучшая WF-модель в среднем по горизонтам (%s): %s (значение=%s)",
            metric_key,
            ob_name,
            ob_meta.get("value"),
        )
    try:
        summary = {
            "selection_metric": metric_key,
            "per_horizon": {
                str(h): results["wf_best"].get(h, {}) for h in horizons
            },
            "per_horizon_clf": {
                str(h): results["wf_best_clf"].get(h, {}) for h in horizons
            },
            "overall_across_horizons": results["wf_best_overall"],
        }
        json_path = Path(out) / "best_models_wf.json"
        json_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("[ML] Сводка лучших WF-моделей: %s", json_path)
    except Exception as exc:
        logger.warning("[ML] сохранение best_models_wf.json: %s", exc)

    try:
        st_rows = []
        for h in horizons:
            st = results["stacking"].get(h)
            if isinstance(st, dict) and st:
                st_rows.append({"horizon": int(h), **st})
        if st_rows:
            pd.DataFrame(st_rows).to_csv(
                Path(out) / "stacking_summary.csv", index=False, encoding="utf-8"
            )
    except Exception as exc:
        logger.warning("[ML] сохранение stacking_summary.csv: %s", exc)

    return results


def _export_news_ablation(
    df_feat: pd.DataFrame, cfg: Dict[str, Any], out: Path
) -> None:
    """
    Доп. запуск ML без новостей и сравнение c основным прогоном.
    """
    if not bool(cfg.get("RUN_NEWS_ABLATION", False)):
        return
    cfg_wo = dict(cfg)
    cfg_wo["USE_NEWS_SENTIMENT"] = False
    logger.info("[ABLATION] Запуск ML без новостных признаков")
    ml_wo = run_ml(df_feat, cfg_wo, out)

    rows: list[dict[str, Any]] = []
    horizons = list(cfg.get("HORIZONS", [1, 5, 10]))
    for h in horizons:
        base = ml_wo.get("wf_best", {}).get(h, {})
        with_news = {}
        path_with = out / f"wf_reg_mean_h{h}.csv"
        path_without = out / f"wf_reg_mean_h{h}_no_news.csv"
        try:
            # сохраним no-news сводку отдельно
            agg_wo = ml_wo.get("wf_reg_agg", {}).get(h)
            if isinstance(agg_wo, pd.DataFrame) and not agg_wo.empty:
                agg_wo.to_csv(path_without, index=False, encoding="utf-8")
            if path_with.exists():
                df_with = pd.read_csv(path_with)
                if "model" in df_with.columns:
                    mname = str(
                        (json.loads((out / "best_models_wf.json").read_text(encoding="utf-8")))
                        .get("per_horizon", {})
                        .get(str(h), {})
                        .get("model", "")
                    )
                    if mname:
                        sub = df_with[df_with["model"] == mname]
                        if not sub.empty:
                            with_news = sub.iloc[0].to_dict()
        except Exception:
            with_news = {}
        row: dict[str, Any] = {"horizon": int(h)}
        mae_with = with_news.get("MAE")
        mae_wo = None
        if isinstance(base, dict):
            best_model_wo = base.get("model", "")
            agg_wo = ml_wo.get("wf_reg_agg", {}).get(h)
            if isinstance(agg_wo, pd.DataFrame) and not agg_wo.empty and best_model_wo:
                sub_wo = agg_wo[agg_wo["model"] == best_model_wo]
                if not sub_wo.empty:
                    mae_wo = float(sub_wo.iloc[0].get("MAE"))
                    row["without_news_model"] = best_model_wo
                    row["without_news_MAE"] = mae_wo
        if mae_with is not None:
            row["with_news_MAE"] = float(mae_with)
        if mae_with is not None and mae_wo is not None and mae_wo != 0:
            row["delta_mae_pct_vs_without"] = float((mae_with - mae_wo) / mae_wo * 100.0)
        rows.append(row)

    try:
        if rows:
            pd.DataFrame(rows).to_csv(
                out / "ablation_news_vs_no_news.csv", index=False, encoding="utf-8"
            )
    except Exception as exc:
        logger.warning("[ABLATION] save ablation_news_vs_no_news.csv: %s", exc)


def run_experiment(
    cfg: Dict[str, Any],
    data_path: str | Path,
    out_dir: str | Path,
) -> Dict[str, Any]:
    """
    Полный эксперимент: загрузка, валидация, индикаторы, EDA, ARIMA–GARCH, ML.

    Returns
    -------
    Словарь с ключами data, arima_garch, ml, out_dir.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("[PIPELINE] Start experiment: data=%s, out=%s", data_path, out)

    df = load_data(data_path)
    df = validate_data(df, cfg)

    df = df.copy()
    df["log_ret"] = compute_log_returns(df["CLOSE"])

    df_feat = compute_indicators(df, cfg)
    df_feat = attach_news_sentiment_features(df_feat, cfg)
    run_eda(df_feat, out, cfg)
    export_eda_stats(df_feat, out, cfg)

    arima_garch_df = run_hybrid_arima_garch(df_feat, cfg, out)
    ml_results = run_ml(df_feat, cfg, out)
    _export_news_ablation(df_feat, cfg, out)

    logger.info("[PIPELINE] Experiment finished")

    return {
        "data": df_feat,
        "arima_garch": arima_garch_df,
        "ml": ml_results,
        "out_dir": str(out),
    }


if __name__ == "__main__":
    from config import CFG

    run_experiment(
        cfg=CFG,
        data_path=CFG.get("FILE_PATH", "OZON_combined.csv"),
        out_dir=CFG.get("OUT_DIR", "./results"),
    )
