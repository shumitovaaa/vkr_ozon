"""
Единая сборка пайплайна: EDA, baseline ARIMA–GARCH, walk-forward ML (Ridge, RF, LGB), стекинг.

Сравнение регрессоров — по единому walk-forward; лучшая модель на горизонт и в среднем по h.

Точка входа: run_experiment().
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

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
from models.walk_forward import (
    rolling_forecast_hybrid_arima_garch,
    walk_forward_train_eval,
)
from models.stacking import fit_stacking
from visualization.plots import Visualizer

logger = logging.getLogger(__name__)


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
    Rolling baseline: μ̂ из ARIMA(p,d,q), σ̂² по ε̂ из GARCH; график h=1.

    Returns
    -------
    DataFrame с колонками y_realized, mu_forecast, ci_lower, ci_upper.
    """
    if "log_ret" not in df.columns:
        df = df.copy()
        df["log_ret"] = np.log(df["CLOSE"] / df["CLOSE"].shift(1))

    series = df["log_ret"].dropna()
    test_frac = cfg.get("TEST_FRACTION", 0.2)
    test_size = int(len(series) * test_frac)
    if cfg.get("ARIMA_MAX_TEST_DAYS"):
        test_size = min(test_size, int(cfg["ARIMA_MAX_TEST_DAYS"]))
    test_size = max(30, test_size)
    logger.info(
        "[ARIMA-GARCH] test_size=%s из %s (%.1f%%)",
        test_size,
        len(series),
        100.0 * test_size / max(len(series), 1),
    )

    order_cfg = cfg.get("ARIMA_ORDER", [1, 0, 1])
    arima_order = tuple(int(x) for x in order_cfg[:3])
    if len(arima_order) < 3:
        arima_order = (1, 0, 1)

    forecast_df = rolling_forecast_hybrid_arima_garch(
        y=series,
        test_size=test_size,
        arima_order=arima_order,
        alpha=cfg.get("CI_ALPHA", 0.10),
        garch_p=int(cfg.get("GARCH_P", 1)),
        garch_q=int(cfg.get("GARCH_Q", 1)),
        min_train_for_hybrid=int(cfg.get("ARIMA_GARCH_MIN_TRAIN", 100)),
        use_garch_on_eps=bool(cfg.get("USE_ARIMA_GARCH", True)),
    )

    viz = Visualizer(out)
    try:
        viz.plot_arima_garch_forecast(
            y_full=series,
            forecast_df=forecast_df,
            horizon=1,
            model_label=cfg.get("ARIMA_PLOT_LABEL", "ARIMA-GARCH"),
        )
    except Exception as exc:
        logger.warning("[ARIMA-GARCH] plot failed: %s", exc)

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
    split_point = int(len(X) * (1 - test_frac))
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
        best_row = df_clf.sort_values("Accuracy", ascending=False).iloc[0]
        best_name = best_row["model"]
        pred_col = f"pred_{best_name}"
        if pred_col in preds_df.columns:
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
        results["stacking"][h] = fit_stacking(
            X_train=X_tr,
            y_train=y_tr.values,
            X_test=X_te,
            y_test=y_te.values,
            cfg=cfg,
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
        "wf_best": {},
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

    return results


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
    df["log_ret"] = np.log(df["CLOSE"] / df["CLOSE"].shift(1))

    df_feat = compute_indicators(df, cfg)
    df_feat = attach_news_sentiment_features(df_feat, cfg)
    run_eda(df_feat, out, cfg)
    arima_garch_df = run_hybrid_arima_garch(df_feat, cfg, out)
    ml_results = run_ml(df_feat, cfg, out)

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
