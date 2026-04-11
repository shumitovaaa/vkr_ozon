"""
Подбор гиперпараметров LightGBM через Optuna (TPE).

Сильные стороны для OZON: автоматический поиск глубины и регуляризации под
временной кросс-валидацией.

Слабые стороны: дорого по времени; при малой выборке оптимум может переоценивать
шум. В основном пайплайне сейчас не вызывается — оставлен для экспериментов.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

logger = logging.getLogger(__name__)


def _random_seed(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("RANDOM_STATE", cfg.get("SEED", 42)))


def optuna_lgb_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Dict[str, Any],
    n_trials: int = 100,
    task: str = "regression",
) -> Dict[str, Any]:
    """
    Оптимизировать гиперпараметры LGB через Optuna и TimeSeriesSplit.

    Parameters
    ----------
    X_train, y_train
        Обучающая матрица и цель.
    cfg
        Должен содержать TEST_FRACTION, OPTUNA_TIMEOUT, RANDOM_STATE/SEED.
    task
        'regression' или 'classification'.

    Returns
    -------
    Словарь лучших параметров или {} если библиотеки недоступны.
    """
    if not (HAS_LGB and HAS_OPTUNA):
        logger.warning("Optuna пропущен: LightGBM или Optuna недоступны.")
        return {}

    rs = _random_seed(cfg)
    n_splits = max(2, int(1 / max(cfg.get("TEST_FRACTION", 0.2), 0.05)))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 5, 40
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            ),
            "random_state": rs,
            "n_jobs": -1,
            "verbose": -1,
        }
        model_cls = (
            lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
        )
        scores: list[float] = []
        for tr_i, va_i in tscv.split(X_train):
            try:
                model = model_cls(**params)
                model.fit(X_train[tr_i], y_train[tr_i])
                yp = model.predict(X_train[va_i])
                if task == "regression":
                    err = mean_absolute_error(y_train[va_i], yp)
                else:
                    err = 1.0 - accuracy_score(y_train[va_i], yp)
                scores.append(err)
            except Exception as exc:
                logger.error("Optuna trial error: %s", exc)
                scores.append(float("inf"))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=rs),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=float(cfg.get("OPTUNA_TIMEOUT", 300)),
    )
    best = dict(study.best_params)
    best.update({"n_jobs": -1, "verbose": -1, "random_state": rs})
    logger.info("Optuna best: %s", best)
    return best
