"""
Единый словарь настроек ``CFG`` и структура ``ExperimentResults``.

Используется в ``main``, ``pipeline``, ``scripts``. Секции: индикаторы,
walk-forward, новости (``NEWS_*``), VK (``VK_*``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

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
    "N_LAGS": 10,
    "OBV_NORM_WINDOW": 50,
    "OBV_WINDOW": 20,
    "USE_LOG_RET": True,
    "ADD_TIME_FEATURES": True,
    "NORMALIZE_OBV": True,
    "WINSORIZE_OUTLIERS": False,
    "OUTLIER_ZSCORE": 3.5,
    "HORIZONS": [1, 5, 10],
    "WF_SPLITS": 10,
    "TEST_FRACTION": 0.2,
    "TRAIN_WINDOW": 504,
    "EMBARGO_DAYS": 10,
    "WALK_FORWARD_TEST_SIZE": None,
    "ARIMA_MAX_TEST_DAYS": None,
    "ARIMA_ORDER": [1, 0, 1],
    "ARIMA_RETRAIN_FREQ": 21,
    "USE_ARIMA_GARCH": True,
    "GARCH_P": 1,
    "GARCH_Q": 1,
    "ARIMA_GARCH_MIN_TRAIN": 100,
    "RANDOM_STATE": 42,
    "SEED": 42,
    "RIDGE_ALPHA": 1.0,
    "RF_N_EST": 200,
    "RF_N_ESTIMATORS": 200,
    "RF_MAX_DEPTH": 8,
    "RF_MAX_DEPTH_REG": 6,
    "RF_MAX_DEPTH_CLF": 5,
    "RF_MIN_SAMPLES": 5,
    "LGB_N_EST": 300,
    "LGB_N_ESTIMATORS": 300,
    "LGB_LR": 0.05,
    "LGB_LEAVES": 31,
    "CI_ALPHA": 0.10,
    # Выбор лучшей WF-модели: MAE, RMSE, MAPE (минимум) или R2 (максимум)
    "BEST_MODEL_METRIC": "RMSE",
    "OPTUNA_TRIALS": 100,
    "OPTUNA_TIMEOUT": 300,
    "LAST_N_BARS": 300,
    "FIG_DPI": 150,
    "FIG_FORMAT": "png",
    "SAVE_FEATURES_CSV": True,
    "OUT_DIR": "./results",
    "FILE_PATH": "OZON_combined.csv",
    "COMMISSION": 0.0005,
    "USE_NEWS_SENTIMENT": True,
    # После scripts/compute_sentiment.py укажите news_filtered_sentiment.csv (см. NEWS_INTEGRATION.md)
    "NEWS_CSV_PATH": "data/news.csv",
    # Идентификатор HF Hub или путь к каталогу с config.json после обучения своей головы
    # (AutoModelForSequenceClassification.save_pretrained). База rubert-tiny2 с Hub без
    # дообучения не подходит — см. news/sentiment.py (модульный docstring).
    # Публичная RU sentiment (при 404 на другой модели смените на cointegrated/rubert-tiny-sentiment-balanced)
    "NEWS_MODEL": "seara/rubert-tiny2-russian-sentiment",
    "NEWS_BATCH_SIZE": 8,
    "NEWS_DEVICE": None,
    # Токен HF для приватных/gated репозиториев; иначе можно задать env HF_TOKEN
    "NEWS_HF_TOKEN": None,
    # Ветка на Hub (по умолчанию main); для локального каталога не используется
    "NEWS_HF_REVISION": None,
    "NEWS_MAX_LENGTH": 512,
    "NEWS_DATE_COL": None,
    "NEWS_TITLE_COL": None,
    "NEWS_BODY_COL": None,
    # После парсинга VK и при load_news_csv — оставить только строки с ключевыми словами Ozon (см. news/ozon_filter.py)
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


@dataclass
class ExperimentResults:
    config: Dict[str, Any] = field(default_factory=dict)
    adf_results: list[Any] = field(default_factory=list)
    arima_garch_mae: float = 0.0
    wf_reg_metrics: list[Any] = field(default_factory=list)
    wf_clf_metrics: list[Any] = field(default_factory=list)
    wf_predictions: Any = None
    stacking_metrics: Dict[str, Any] = field(default_factory=dict)
    optuna_params: Dict[str, Any] = field(default_factory=dict)
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    buy_hold: Dict[str, Any] = field(default_factory=dict)
    horizon_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        payload = asdict(self)
        payload.pop("wf_predictions", None)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        logger.info("ExperimentResults сохранены: %s", path)
        if self.wf_predictions is not None:
            pq = path.with_suffix(".predictions.csv")
            self.wf_predictions.to_csv(pq)
            logger.info("WF-предсказания: %s", pq)
