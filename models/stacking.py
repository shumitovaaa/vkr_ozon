"""
Двухуровневый стекинг: предсказания RF и LGB → мета-Ridge.

Сильные стороны для OZON: комбинирует «гладкость» леса и градиентный бустинг.

Слабые стороны: мета-уровень линейный; при смене режима рынка веса мета-модели
могут устареть; внутренний сплит 80/20 на train уменьшает данные для базовых
моделей.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from evaluation.metrics import regression_metrics
from models.scaling import robust_scale_train_test


def fit_stacking(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Обучить RF+LGB на первых 80% train, мета-Ridge на остальных 20%.

    Parameters
    ----------
    X_train, y_train
        Обучающая выборка hold-out блока пайплайна.
    X_test, y_test
        Финальный тест для отчёта метрик.

    Returns
    -------
    Словарь метрик регрессии для метки Stacking(RF+LGB).
    """
    seed = cfg.get("SEED", cfg.get("RANDOM_STATE", 42))

    Xs_tr, Xs_te = robust_scale_train_test(
        X_train,
        X_test,
        winsorize=bool(cfg.get("WINSORIZE_OUTLIERS", False)),
    )

    split = int(len(Xs_tr) * 0.8)

    rf = RandomForestRegressor(
        n_estimators=int(cfg.get("RF_N_EST_REG", 200)),
        max_depth=int(cfg.get("RF_MAX_DEPTH_REG", 6)),
        min_samples_leaf=int(cfg.get("RF_MIN_SAMPLES_LEAF_REG", 5)),
        random_state=seed,
        n_jobs=-1,
    )
    lgb = LGBMRegressor(
        n_estimators=int(cfg.get("LGB_N_EST", 300)),
        learning_rate=float(cfg.get("LGB_LR", 0.05)),
        num_leaves=int(cfg.get("LGB_LEAVES", 31)),
        max_depth=int(cfg.get("LGB_MAX_DEPTH", -1)),
        min_child_samples=int(cfg.get("LGB_MIN_CHILD_SAMPLES", 20)),
        subsample=float(cfg.get("LGB_SUBSAMPLE", 0.8)),
        colsample_bytree=float(cfg.get("LGB_COLSAMPLE_BYTREE", 0.8)),
        verbose=-1,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(Xs_tr.iloc[:split], y_train[:split])
    lgb.fit(Xs_tr.iloc[:split], y_train[:split])

    meta_tr = np.column_stack(
        [
            rf.predict(Xs_tr.iloc[split:]),
            lgb.predict(Xs_tr.iloc[split:]),
        ]
    )
    meta_ridge = Ridge(alpha=float(cfg.get("RIDGE_ALPHA", 1.0)))
    meta_ridge.fit(meta_tr, y_train[split:])

    meta_te = np.column_stack([rf.predict(Xs_te), lgb.predict(Xs_te)])
    pred = meta_ridge.predict(meta_te)

    return regression_metrics(y_test, pred, "Stacking(RF+LGB)")
