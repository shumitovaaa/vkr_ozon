"""
Walk-forward валидация и rolling baseline ARIMA–GARCH на лог-доходностях OZON.

Модели и релевантность для OZON (волатильный ряд, структурные сдвиги):

- Ridge: быстрая линейная база, хорошо при малом числе наблюдений на фолд;
  слабость — не улавливает нелинейные эффекты разворотов и режимов.

- RandomForest (reg/clf): устойчив к выбросам, интерпретируемость через важности;
  на шумных доходностях склонен к переобучению при глубоких деревьях.

- LightGBM (reg/clf): сильная нелинейность и взаимодействия признаков;
  чувствителен к утечке масштаба — здесь снято RobustScaler по train.

- Гибрид ARIMA–GARCH: среднее оценивается ARIMA(p,d,q) (statsmodels), на
  in-sample остатках ARIMA оценивается GARCH(p,q) с нулевым средним (arch).
  Прогноз на 1 шаг: μ из ARIMA, ширина ДИ — из прогнозной условной дисперсии
  GARCH для следующей инновации (нормальное приближение). Если arch не
  сходится или мало остатков — ДИ берётся из statsmodels ARIMA.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA

from evaluation.metrics import (
    classification_metrics,
    regression_metrics,
    validate_residuals,
)
from models.scaling import robust_scale_train_test

try:
    from arch import arch_model

    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

logger = logging.getLogger(__name__)

TRAIN_WINDOW = 252 * 2
EMBARGO_DAYS = 10

# Имена колонок прогнозной таблицы (y_t — факт, μ̂ — условное среднее ARIMA)
COL_Y_REALIZED = "y_realized"
COL_MU_FORECAST = "mu_forecast"
COL_CI_LOWER = "ci_lower"
COL_CI_UPPER = "ci_upper"


def _build_models(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Создать свежие экземпляры моделей под один фолд."""
    seed = cfg.get("SEED", cfg.get("RANDOM_STATE", 42))
    lgb_params = {
        "n_estimators": cfg.get("LGB_N_EST", 300),
        "learning_rate": cfg.get("LGB_LR", 0.05),
        "num_leaves": cfg.get("LGB_LEAVES", 31),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "verbose": -1,
    }
    return {
        "Ridge": Ridge(alpha=cfg.get("RIDGE_ALPHA", 1.0)),
        "RFreg": RandomForestRegressor(
            n_estimators=cfg.get("RF_N_EST", 200),
            max_depth=cfg.get("RF_MAX_DEPTH", 8),
            random_state=seed,
            n_jobs=-1,
        ),
        "LGBreg": LGBMRegressor(**lgb_params),
        "RFclf": RandomForestClassifier(
            n_estimators=cfg.get("RF_N_EST", 200),
            max_depth=cfg.get("RF_MAX_DEPTH", 8),
            random_state=seed,
            n_jobs=-1,
        ),
        "LGBclf": LGBMClassifier(**lgb_params),
    }


def _purged_splits(
    n: int,
    n_splits: int = 5,
    embargo: int = EMBARGO_DAYS,
    train_window: int = TRAIN_WINDOW,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Скользящие train/test с embargo между концом train и началом test.

    Доля теста суммарно ~20% выборки, делится на n_splits фолдов.
    """
    total_test = int(n * 0.20)
    test_size = max(1, total_test // n_splits)

    for i in range(n_splits):
        test_end = n - (n_splits - 1 - i) * test_size
        test_start = test_end - test_size + embargo
        train_end = test_start - embargo
        train_start = max(0, train_end - train_window)

        if train_end <= train_start or test_start >= test_end:
            continue

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx


def _crisis_sample_weights(X_tr: pd.DataFrame) -> np.ndarray:
    """Снизить вес кризисных баров 2022 — меньше тянуть модель к прошлому режиму."""
    crisis = X_tr.get("in_crisis", pd.Series(0, index=X_tr.index))
    return np.where(crisis == 1, 0.3, 1.0)


def _fit_regressors_on_fold(
    models: Dict[str, Any],
    Xs_tr: pd.DataFrame,
    Xs_te: pd.DataFrame,
    y_r_tr: pd.Series,
    y_r_te: pd.Series,
    sample_w: np.ndarray,
    fold: int,
    horizon: int,
    fold_preds: pd.DataFrame,
    res_reg: List[Dict[str, Any]],
) -> None:
    """Обучить Ridge/RF/LGB регрессоры на одном фолде, записать метрики."""
    for name in ["Ridge", "RFreg", "LGBreg"]:
        model = models[name]
        try:
            if name in ("RFreg", "LGBreg"):
                model.fit(Xs_tr, y_r_tr, sample_weight=sample_w)
            else:
                model.fit(Xs_tr, y_r_tr)
            pred = model.predict(Xs_te)
            fold_preds[f"pred_{name}"] = pred

            metrics = regression_metrics(y_r_te.values, pred, name)
            metrics["fold"] = fold
            metrics["horizon"] = horizon
            res_reg.append(metrics)
        except Exception as exc:
            logger.error("[WF] fold=%s %s failed: %s", fold, name, exc)


def _fit_classifiers_on_fold(
    models: Dict[str, Any],
    Xs_tr: pd.DataFrame,
    Xs_te: pd.DataFrame,
    y_c_tr: pd.Series,
    y_c_te: pd.Series,
    sample_w: np.ndarray,
    fold: int,
    horizon: int,
    fold_preds: pd.DataFrame,
    res_clf: List[Dict[str, Any]],
) -> None:
    """Обучить RF/LGB классификаторы на одном фолде."""
    for name in ["RFclf", "LGBclf"]:
        model = models[name]
        try:
            if name == "LGBclf":
                model.fit(Xs_tr, y_c_tr, sample_weight=sample_w)
            else:
                model.fit(Xs_tr, y_c_tr)
            pred_clf = model.predict(Xs_te)
            pred_prob = model.predict_proba(Xs_te)[:, 1]
            fold_preds[f"pred_{name}"] = pred_clf

            metrics = classification_metrics(
                y_c_te.values, pred_clf, pred_prob, name
            )
            metrics["fold"] = fold
            metrics["horizon"] = horizon
            res_clf.append(metrics)
        except Exception as exc:
            logger.error("[WF] fold=%s %s failed: %s", fold, name, exc)


def walk_forward_train_eval(
    X: pd.DataFrame,
    y_reg: pd.Series,
    y_clf: pd.Series,
    cfg: Dict[str, Any],
    horizon: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame]:
    """
    Purged walk-forward: метрики регрессии/классификации и таблица предсказаний.

    Returns
    -------
    res_reg, res_clf, preds_df
        Списки словарей метрик по фолдам и склеенный DataFrame прогнозов.
    """
    models = _build_models(cfg)
    n = len(X)
    n_splits = cfg.get("WF_SPLITS", 5)

    res_reg: List[Dict[str, Any]] = []
    res_clf: List[Dict[str, Any]] = []
    all_preds: List[pd.DataFrame] = []

    for fold, (tr_idx, te_idx) in enumerate(
        _purged_splits(
            n,
            n_splits,
            embargo=int(cfg.get("EMBARGO_DAYS", EMBARGO_DAYS)),
            train_window=int(cfg.get("TRAIN_WINDOW", TRAIN_WINDOW)),
        )
    ):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_r_tr, y_r_te = y_reg.iloc[tr_idx], y_reg.iloc[te_idx]
        y_c_tr, y_c_te = y_clf.iloc[tr_idx], y_clf.iloc[te_idx]

        Xs_tr, Xs_te = robust_scale_train_test(X_tr, X_te)
        sample_w = _crisis_sample_weights(X_tr)

        fold_preds = pd.DataFrame(
            {"y_true_reg": y_r_te.values, "y_true_clf": y_c_te.values},
            index=X_te.index,
        )

        _fit_regressors_on_fold(
            models,
            Xs_tr,
            Xs_te,
            y_r_tr,
            y_r_te,
            sample_w,
            fold,
            horizon,
            fold_preds,
            res_reg,
        )
        _fit_classifiers_on_fold(
            models,
            Xs_tr,
            Xs_te,
            y_c_tr,
            y_c_te,
            sample_w,
            fold,
            horizon,
            fold_preds,
            res_clf,
        )

        all_preds.append(fold_preds)
        logger.info(
            "[WF] h=%s fold=%s/%s train=%s test=%s",
            horizon,
            fold + 1,
            n_splits,
            len(tr_idx),
            len(te_idx),
        )

    preds_df = pd.concat(all_preds).sort_index()

    for name in ["Ridge", "RFreg", "LGBreg"]:
        col = f"pred_{name}"
        if col in preds_df.columns:
            residuals = (preds_df["y_true_reg"] - preds_df[col]).dropna()
            report = validate_residuals(residuals.values, name)
            logger.info("[RESIDUALS] %s: %s", name, report)

    return res_reg, res_clf, preds_df


def _one_step_arima_mean_and_gaussian_ci(
    y_history: pd.Series,
    arima_order: Tuple[int, int, int],
    alpha: float,
) -> Tuple[float, float, float]:
    """
    Только уравнение среднего ARIMA(p,d,q): μ̂_{t+1} и гауссов ДИ из statsmodels.
    """
    arima_res = ARIMA(y_history, order=arima_order).fit()
    fc = arima_res.get_forecast(steps=1)
    mu_hat = float(fc.predicted_mean.iloc[0])
    ci = fc.conf_int(alpha=alpha)
    return mu_hat, float(ci.iloc[0, 0]), float(ci.iloc[0, 1])


def _one_step_hybrid_arima_mean_garch_variance_ci(
    y_history: pd.Series,
    arima_order: Tuple[int, int, int],
    garch_p: int,
    garch_q: int,
    alpha: float,
    *,
    min_eps_for_garch: int = 40,
) -> Tuple[float, float, float]:
    """
    Гибрид: μ̂ из ARIMA; на остатках ε̂_t оценивается GARCH → σ̂²_{t+1|t}.

    ДИ: μ̂ ± z_{1-α/2}·σ̂. При сбое GARCH — ДИ из ARIMA.
    Остатки ×100 внутри arch, дисперсия возвращается в шкале y_history.
    """
    arima_res = ARIMA(y_history, order=arima_order).fit()
    fc = arima_res.get_forecast(steps=1)
    mu_hat = float(fc.predicted_mean.iloc[0])
    ci_arima = fc.conf_int(alpha=alpha)
    ci_lo_arima = float(ci_arima.iloc[0, 0])
    ci_hi_arima = float(ci_arima.iloc[0, 1])

    eps_hat = arima_res.resid.dropna().values.astype(np.float64)
    n_min = max(int(min_eps_for_garch), int(garch_p) + int(garch_q) + 15)

    if not HAS_ARCH or len(eps_hat) < n_min:
        return mu_hat, ci_lo_arima, ci_hi_arima

    eps_scaled = eps_hat * 100.0
    try:
        garch_spec = arch_model(
            eps_scaled,
            mean="Zero",
            vol="GARCH",
            p=int(garch_p),
            q=int(garch_q),
            rescale=False,
        )
        garch_res = garch_spec.fit(disp="off", show_warning=False)
        garch_fc = garch_res.forecast(horizon=1, reindex=False)
        sigma2_scaled = float(garch_fc.variance.values[-1, 0])
        sigma2_hat = max(sigma2_scaled / 10000.0, 1e-16)
        z_crit = float(norm.ppf(1.0 - alpha / 2.0))
        half_width = z_crit * np.sqrt(sigma2_hat)
        return mu_hat, mu_hat - half_width, mu_hat + half_width
    except Exception:
        return mu_hat, ci_lo_arima, ci_hi_arima


def rolling_forecast_hybrid_arima_garch(
    y: pd.Series,
    test_size: int,
    arima_order: Tuple[int, int, int] = (1, 0, 1),
    alpha: float = 0.10,
    *,
    garch_p: int = 1,
    garch_q: int = 1,
    min_train_for_hybrid: int = 100,
    use_garch_on_eps: bool = True,
) -> pd.DataFrame:
    """
    Expanding-window прогноз на 1 шаг: гибрид ARIMA (μ) + GARCH(σ²|ε̂) либо только ARIMA.

    Parameters
    ----------
    y
        Ряд лог-доходностей (стационарный; индекс — даты).
    arima_order
        (p, d, q) для уравнения среднего ARIMA(p,d,q).
    alpha
        Уровень значимости для ДИ (совместный с CI_ALPHA в CFG).
    use_garch_on_eps
        Если False — только ARIMA для μ̂ и доверительного интервала.
    """
    y = y.dropna()
    n = len(y)
    train_end = n - test_size

    mu_forecasts: List[float] = []
    y_realized: List[float] = []
    ci_lower: List[float] = []
    ci_upper: List[float] = []
    idx_test: List[Any] = []

    if use_garch_on_eps and not HAS_ARCH:
        logger.warning(
            "[ARIMA-GARCH] пакет arch не установлен — только блок среднего ARIMA."
        )

    for step in range(test_size):
        y_history = y.iloc[: train_end + step]
        mu_hat = 0.0
        lo = 0.0
        hi = 0.0
        hybrid_ok = False

        if use_garch_on_eps and HAS_ARCH and len(y_history) >= min_train_for_hybrid:
            try:
                mu_hat, lo, hi = _one_step_hybrid_arima_mean_garch_variance_ci(
                    y_history,
                    arima_order=arima_order,
                    garch_p=garch_p,
                    garch_q=garch_q,
                    alpha=alpha,
                    min_eps_for_garch=max(40, garch_p + garch_q + 15),
                )
                hybrid_ok = True
            except Exception as exc:
                logger.warning(
                    "[ARIMA-GARCH] step=%s гибрид: %s — fallback только ARIMA",
                    step,
                    exc,
                )

        if not hybrid_ok:
            try:
                mu_hat, lo, hi = _one_step_arima_mean_and_gaussian_ci(
                    y_history, arima_order, alpha
                )
            except Exception as exc:
                logger.warning("[ARIMA] step=%s: %s", step, exc)
                mu_hat, lo, hi = 0.0, 0.0, 0.0

        mu_forecasts.append(mu_hat)
        ci_lower.append(lo)
        ci_upper.append(hi)
        y_realized.append(float(y.iloc[train_end + step]))
        idx_test.append(y.index[train_end + step])

    return pd.DataFrame(
        {
            COL_Y_REALIZED: y_realized,
            COL_MU_FORECAST: mu_forecasts,
            COL_CI_LOWER: ci_lower,
            COL_CI_UPPER: ci_upper,
        },
        index=pd.Index(idx_test, name="date"),
    )
