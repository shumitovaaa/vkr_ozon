"""
Единый словарь настроек ``CFG``: данные, горизонты, WF, индикаторы, новости, параметры моделей.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Значения по умолчанию для путей и горизонтов (единый источник для CFG и fallback в коде)
DEFAULT_FILE_PATH = "OZON_combined.csv"
DEFAULT_OUT_DIR = "./results"
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10)

def resolve_news_features_flag(cfg: Dict[str, Any]) -> bool:
    """
    Единая точка получения значения переключателя «использовать ли новости».

    Приоритет: ``USE_NEWS_FEATURES`` → ``USE_NEWS_SENTIMENT``. Любая правда
    в любом из ключей включает новости (False имеет приоритет, только если
    оба ключа явно False). Это убирает риск того, что ``run_experiment``
    выставит один ключ, а потребитель проверит другой и наоборот.

    Returns
    -------
    bool
        True — собирать и подключать новостные признаки; False — baseline.
    """
    has_canonical = "USE_NEWS_FEATURES" in cfg
    has_legacy = "USE_NEWS_SENTIMENT" in cfg
    if has_canonical and has_legacy:
        return bool(cfg.get("USE_NEWS_FEATURES")) and bool(cfg.get("USE_NEWS_SENTIMENT"))
    if has_canonical:
        return bool(cfg.get("USE_NEWS_FEATURES"))
    return bool(cfg.get("USE_NEWS_SENTIMENT", False))


def set_news_features_flag(cfg: Dict[str, Any], value: bool) -> Dict[str, Any]:
    """Согласованно проставить оба ключа (канонический + алиас); cfg-копия не делается."""
    cfg["USE_NEWS_FEATURES"] = bool(value)
    cfg["USE_NEWS_SENTIMENT"] = bool(value)
    return cfg


CFG: Dict[str, Any] = {
    "SMA_PERIODS": [20, 50, 200],
    "RSI_PERIOD": 14,
    "STOCH_PERIOD": 14,
    "STOCH_SMOOTH": 3,
    "BB_PERIOD": 20,
    "BB_NBDEV": 2,
    "ATR_PERIOD": 14,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    # Лаги ln(C_t/C_{t-1}) как log_ret_lag1..log_ret_lagN в индикаторах и в X
    "N_LAGS": 10,
    # Окно rolling mean/std для OBV_z (если не задано — используется OBV_WINDOW)
    "OBV_NORM_WINDOW": 50,
    "OBV_WINDOW": 20,
    "USE_LOG_RET": True,
    "ADD_TIME_FEATURES": True,
    "NORMALIZE_OBV": True,
    # Винзоризация признаков по квантилям train (5%/95%) перед RobustScaler в WF/стекинге
    "WINSORIZE_OUTLIERS": False,
    "OUTLIER_ZSCORE": 3.5,
    "HORIZONS": list(DEFAULT_HORIZONS),
    "WF_SPLITS": 10,
    "TEST_FRACTION": 0.2,
    "TRAIN_WINDOW": 504,
    "EMBARGO_DAYS": 10,
    "ARIMA_MAX_TEST_DAYS": None,
    # Горизонты rolling ARIMA(-GARCH): h=1 — гибрид с GARCH; h>1 — кумулятивный прогноз
    "ARIMA_FORECAST_HORIZONS": list(DEFAULT_HORIZONS),
    # Метрика для выбора лучшего h на графике ARIMA (None = как BEST_MODEL_METRIC)
    "ARIMA_BEST_HORIZON_METRIC": None,
    # Выбор лучшей WF-модели: MAE, RMSE, MAPE (минимум) или R2 (максимум)
    "BEST_MODEL_METRIC": "RMSE",
    "RANDOM_STATE": 42,
    "SEED": 42,
    # --- ARIMA–GARCH ---
    "ARIMA_ORDER": [1, 0, 1],
    "GARCH_P": 1,
    "GARCH_Q": 1,
    "ARIMA_GARCH_MIN_TRAIN": 100,
    "USE_ARIMA_GARCH": True,
    "CI_ALPHA": 0.10,
    # --- Ridge / RandomForest / LightGBM ---
    "RIDGE_ALPHA": 1.0,
    "RF_N_EST_REG": 200,
    "RF_MAX_DEPTH_REG": 6,
    "RF_MIN_SAMPLES_LEAF_REG": 5,
    "RF_N_EST_CLF": 200,
    "RF_MAX_DEPTH_CLF": 5,
    "RF_MIN_SAMPLES_LEAF_CLF": 5,
    "LGB_N_EST": 300,
    "LGB_LR": 0.05,
    "LGB_LEAVES": 31,
    "LGB_MAX_DEPTH": -1,
    "LGB_MIN_CHILD_SAMPLES": 20,
    "LGB_SUBSAMPLE": 0.8,
    "LGB_COLSAMPLE_BYTREE": 0.8,
    "LGBCLF_N_EST": None,
    "LGBCLF_LR": None,
    "LGBCLF_LEAVES": None,
    "LGBCLF_MAX_DEPTH": None,
    "LGBCLF_MIN_CHILD_SAMPLES": None,
    "LGBCLF_SUBSAMPLE": None,
    "LGBCLF_COLSAMPLE_BYTREE": None,
    "OUT_DIR": DEFAULT_OUT_DIR,
    "FILE_PATH": DEFAULT_FILE_PATH,
    "COMMISSION": 0.0005,
    # Канонический переключатель A/B: True = news-enhanced, False = baseline без новостей.
    # Алиас USE_NEWS_SENTIMENT сохранён ради обратной совместимости — при старте
    # пайплайн синхронизирует значения (см. config.resolve_news_features_flag).
    "USE_NEWS_FEATURES": True,
    "USE_NEWS_SENTIMENT": True,
    # Включает A/B сравнение в одном запуске: pipeline.run_ab_comparison делает два
    # прогона с одинаковыми сплитами и сохраняет таблицу с Δ-метриками
    # (см. ab_comparison_*.csv в каталоге результатов).
    "RUN_NEWS_ABLATION": False,
    # Подкаталоги для A/B (относительно OUT_DIR).
    "AB_BASELINE_SUBDIR": "baseline_no_news",
    "AB_NEWS_SUBDIR": "news_enhanced",
    "NEWS_CSV_PATH": "data/news.csv",
    "NEWS_MODEL": "seara/rubert-tiny2-russian-sentiment",
    "NEWS_BATCH_SIZE": 8,
    "NEWS_DEVICE": None,
    "NEWS_HF_TOKEN": None,
    "NEWS_HF_REVISION": None,
    "NEWS_MAX_LENGTH": 512,
    "NEWS_DATE_COL": None,
    "NEWS_TITLE_COL": None,
    "NEWS_BODY_COL": None,
    "NEWS_POST_PARSE_OZON_FILTER": True,
    "NEWS_ONLY_PRICE_DATES": True,
    "NEWS_ALIGN_BOUNDS_TO_CALENDAR": True,
    "NEWS_PRICE_CALENDAR_PATH": "",
    "NEWS_KEYWORDS": [
        "OZON",
        "Ozon",
        "ozon",
        "ozon.ru",
        "OZON.ru",
        "Ozon Group",
        "OZON Group",
        "Ozon Holdings",
        "Ozon Fresh",
        "Ozon Bank",
        "Ozon-Bank",
        "Ozon Logistics",
        "Ozon Invest",
        "ОЗОН",
        "Озон",
        "озон",
        "Группа Ozon",
        "группа Ozon",
        "Группа OZON",
        "Озон-банк",
        "Озон банк",
        "озон-банк",
        "Озон-инвест",
        "маркетплейс Ozon",
        "маркетплейса Ozon",
        "маркетплейс OZON",
        "маркетплейсу Ozon",
        "акции Ozon",
        "акций Ozon",
        "бумаг Ozon",
        "бумаги Ozon",
        "IPO Ozon",
        "ПВЗ Ozon",
        "пункты выдачи Ozon",
        "интернет-ритейлер Ozon",
        "ритейлер Ozon",
        "Ozon и Wildberries",
        "Wildberries и Ozon",
    ],
    "NEWS_MAX_ARTICLES": None,
    "NEWS_SINCE": None,
    "NEWS_UNTIL": None,
    "VK_ACCESS_TOKEN": None,
    "VK_OWNER_ID": None,
    "VK_GROUP_DOMAIN": "",
    "VK_API_VERSION": "5.131",
    "VK_REQUEST_DELAY_SEC": 0.35,
    "VK_CONNECT_TIMEOUT_SEC": 25.0,
    "VK_REQUEST_TIMEOUT_SEC": 60.0,
    "VK_REQUEST_RETRIES": 8,
    "VK_WALL_MAX_POSTS_SCAN": 150_000,
}
