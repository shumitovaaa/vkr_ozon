"""
Графики EDA, walk-forward, ARIMA–GARCH (matplotlib, backend Agg).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import matplotlib

matplotlib.use("Agg")  # без GUI: до pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from models.walk_forward import (
    COL_CI_LOWER,
    COL_CI_UPPER,
    COL_MU_FORECAST,
    COL_Y_REALIZED,
)

logger = logging.getLogger(__name__)


class Visualizer:
    """Сохранение фигур в каталог результатов."""

    def __init__(self, out_dir: Union[str, Path]) -> None:
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def _save(self, filename: str) -> Path:
        path = self.out / filename
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info(f"[PLOT] Сохранён: {path}")
        return path

    # ─── EDA ────────────────────────────────────────────────────────────────

    def plot_price_volume(self, df: pd.DataFrame) -> Path:
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        axes[0].plot(df.index, df["CLOSE"], lw=0.9, color="steelblue")
        axes[0].set_title("Цена закрытия OZON (MOEX)")
        axes[0].set_ylabel("Цена, руб.")
        axes[0].grid(alpha=0.3)

        axes[1].bar(df.index, df["VOL"], color="gray", alpha=0.5, width=1)
        axes[1].set_title("Объём торгов")
        axes[1].set_ylabel("Объём")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        return self._save("fig_price_volume.png")

    def plot_returns(self, df: pd.DataFrame) -> Path:
        if "log_ret" not in df.columns:
            df = df.copy()
            df["log_ret"] = np.log(df["CLOSE"] / df["CLOSE"].shift(1))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(df.index, df["log_ret"], lw=0.6, color="steelblue")
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].set_title("Лог-доходность OZON")
        axes[0].set_ylabel("log-return")
        axes[0].grid(alpha=0.3)

        axes[1].hist(df["log_ret"].dropna(), bins=80, color="steelblue",
                     alpha=0.7, density=True, edgecolor="none")
        axes[1].set_title("Распределение лог-доходности")
        axes[1].set_xlabel("log-return")
        axes[1].grid(alpha=0.3)
        # Нормальное распределение поверх
        mu, sigma = df["log_ret"].dropna().mean(), df["log_ret"].dropna().std()
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
        axes[1].plot(x, stats.norm.pdf(x, mu, sigma), color="red",
                     lw=1.5, label="Норм. распр.")
        axes[1].legend()

        plt.tight_layout()
        return self._save("fig_returns.png")

    def plot_seasonality(self, df: pd.DataFrame) -> Path:
        if "log_ret" not in df.columns:
            df = df.copy()
            df["log_ret"] = np.log(df["CLOSE"] / df["CLOSE"].shift(1))

        DOW_LABELS = {
            0: "ПН", 1: "ВТ", 2: "СР", 3: "ЧТ",
            4: "ПТ", 5: "СБ", 6: "ВС"
        }
        MONTH_LABELS = {
            1: "Янв", 2: "Фев", 3: "Мар", 4: "Апр",
            5: "Май", 6: "Июн", 7: "Июл", 8: "Авг",
            9: "Сен", 10: "Окт", 11: "Ноя", 12: "Дек"
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        dow_mean = df.groupby(df.index.dayofweek)["log_ret"].mean() * 100
        labels_dow = [DOW_LABELS.get(i, str(i)) for i in dow_mean.index]
        colors_dow = ["#d62728" if v < 0 else "#2ca02c" for v in dow_mean.values]
        axes[0].bar(labels_dow, dow_mean.values, color=colors_dow)
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].set_title("Средняя доходность по дням недели, %")
        axes[0].set_ylabel("Лог-доходность, %")
        axes[0].grid(axis="y", alpha=0.3)

        month_mean = df.groupby(df.index.month)["log_ret"].mean() * 100
        labels_m = [MONTH_LABELS.get(i, str(i)) for i in month_mean.index]
        colors_m = ["#d62728" if v < 0 else "#2ca02c" for v in month_mean.values]
        axes[1].bar(labels_m, month_mean.values, color=colors_m)
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_title("Средняя доходность по месяцам, %")
        axes[1].set_ylabel("Лог-доходность, %")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return self._save("fig_seasonality.png")

    def plot_correlation(self, df: pd.DataFrame) -> Path:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Ограничиваем до разумного числа колонок
        num_cols = [c for c in num_cols if c not in ["VOL"]][:25]

        corr_p = df[num_cols].corr(method="pearson")
        corr_s = df[num_cols].corr(method="spearman")

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        for ax, corr, title in zip(
            axes, [corr_p, corr_s],
            ["Пирсон", "Спирмен"]
        ):
            im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
            ax.set_xticks(range(len(num_cols)))
            ax.set_yticks(range(len(num_cols)))
            ax.set_xticklabels(num_cols, rotation=90, fontsize=7)
            ax.set_yticklabels(num_cols, fontsize=7)
            ax.set_title(f"Корреляция ({title})")
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        return self._save("fig_correlation.png")

    def plot_news_sentiment(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Дневная тональность после RuBERT (агрегат по новостям) и число новостей.
        Сохраняет ``fig_news_sentiment.png`` в каталог результатов.
        """
        need = {"sentiment_score", "news_count", "has_news"}
        if not need.issubset(df.columns):
            logger.info("[PLOT] Нет колонок новостей — пропуск fig_news_sentiment")
            return None

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        idx = df.index
        has = df["has_news"].to_numpy() > 0

        axes[0].plot(
            idx,
            df["sentiment_score"],
            color="lightgray",
            lw=0.9,
            label="sentiment_score",
        )
        axes[0].scatter(
            idx[has],
            df.loc[has, "sentiment_score"],
            color="darkgreen",
            s=10,
            zorder=3,
            label="дни с новостями",
        )
        axes[0].axhline(0.0, color="black", lw=0.6)
        axes[0].set_ylabel("Тональность (−1 / 0 / +1)")
        axes[0].set_title("Новости: дневная тональность (RuBERT)")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[0].grid(alpha=0.3)

        axes[1].bar(idx, df["news_count"], width=1.0, color="steelblue", alpha=0.75)
        axes[1].set_ylabel("Число новостей")
        axes[1].set_xlabel("Дата")
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return self._save("fig_news_sentiment.png")

    def plot_acf_pacf(self, series: pd.Series, name: str) -> Path:
        series = series.dropna()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        plot_acf(series, lags=40, ax=axes[0], title=f"ACF — {name}")
        plot_pacf(series, lags=40, ax=axes[1], title=f"PACF — {name}", method="ywm")
        axes[0].grid(alpha=0.3)
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        return self._save(f"fig_acf_{name.replace(' ', '_')}.png")

    # ─── Гибрид ARIMA–GARCH (rolling μ̂, ДИ из σ̂²) ───────────────────────────

    def plot_arima_garch_forecast(
        self,
        y_full: pd.Series,
        forecast_df: pd.DataFrame,
        horizon: int = 1,
        model_label: str = "ARIMA-GARCH",
    ) -> Path | None:
        """
        y_train / тест: факт y_realized, прогноз μ̂, ДИ; снизу y − μ̂.

        forecast_df — выход rolling_forecast_hybrid_arima_garch.
        """
        if forecast_df is None or forecast_df.empty:
            logger.warning("[PLOT] plot_arima_garch_forecast: пустой forecast_df")
            return None

        test_start = forecast_df.index[0]
        y_train = y_full[y_full.index < test_start]
        y_test = forecast_df[COL_Y_REALIZED]
        test_dates = forecast_df.index

        fig, axes = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax = axes[0]

        ax.plot(
            y_train.index, y_train.values,
            lw=1.0, color="#1f77b4",
            label="Обучающая выборка", zorder=2,
        )

        ax.fill_between(
            test_dates,
            forecast_df[COL_CI_LOWER].values,
            forecast_df[COL_CI_UPPER].values,
            alpha=0.25, color="orange",
            label="90% ДИ", zorder=1,
        )

        ax.plot(
            test_dates, y_test.values,
            lw=1.2, color="#ff7f0e",
            label="Факт (тест)", zorder=3,
        )

        ax.plot(
            test_dates, forecast_df[COL_MU_FORECAST].values,
            lw=1.2, color="#d62728", linestyle="--",
            label=rf"$\hat{{\mu}}$ ({model_label})", zorder=4,
        )

        ax.axvline(
            test_start, color="black", lw=1.5, linestyle=":",
            label="Начало теста", zorder=5,
        )
        ax.axvspan(test_start, test_dates[-1], alpha=0.05, color="orange")

        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax.set_title(
            f"{model_label} | Горизонт h={horizon} | "
            f"Прогноз лог-доходности OZON",
            fontsize=12,
        )
        ax.set_ylabel("Лог-доходность")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.25)

        eps_hat = y_test.values - forecast_df[COL_MU_FORECAST].values
        axes[1].bar(
            test_dates, eps_hat,
            color=["#d62728" if r < 0 else "#2ca02c" for r in eps_hat],
            width=1.0, alpha=0.7,
        )
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_title(r"Остатки прогноза ($y - \hat{\mu}$) на тесте")
        axes[1].set_ylabel(r"$y - \hat{\mu}$")
        axes[1].grid(alpha=0.25)

        plt.tight_layout()
        return self._save(f"fig_forecast_arima_garch_h{horizon}.png")
    # ─── Walk-Forward ────────────────────────────────────────────────────────

    def plot_wf_metrics(
        self, df_metrics: pd.DataFrame, horizon: int = 1
    ) -> Optional[Path]:
        if df_metrics.empty:
            return None

        models = df_metrics["model"].unique()
        metric_cols = [c for c in ["MAE", "RMSE", "MAPE", "MDA_%", "R2", "IC"]
                       if c in df_metrics.columns]

        n_metrics = len(metric_cols)
        if n_metrics == 0:
            return None

        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metric_cols):
            for model in models:
                sub = df_metrics[df_metrics["model"] == model]
                ax.plot(sub["fold"].values, sub[metric].values,
                        marker="o", label=model, lw=1.5)
            ax.set_title(f"{metric} (h={horizon})")
            ax.set_xlabel("Фолд")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        plt.suptitle(f"Walk-Forward метрики: горизонт h={horizon}", y=1.02)
        plt.tight_layout()
        return self._save(f"fig_wf_metrics_h{horizon}.png")
    
    def plot_wf_forecast(
        self,
        train_dates: pd.Index,
        train_true: np.ndarray,
        test_dates: pd.Index,
        test_true: np.ndarray,
        test_pred: np.ndarray,
        model_name: str,
        horizon: int,
    ) -> Path:
        """
        Обучение + тест: факт, прогноз, линия старта теста, остатки снизу.

        Стиль согласован с plot_arima_garch_forecast (цвета, подписи).
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                                gridspec_kw={"height_ratios": [3, 1]})

        # ─── Верхний график: обучение + тест + прогноз ──────────────────────────
        ax = axes[0]

        # Обучающая выборка (сплошная синяя)
        ax.plot(train_dates, train_true, lw=1.0, color="#1f77b4",
            label="Обучающая выборка", zorder=2)

        # Тестовая выборка (сплошная оранжевая)
        ax.plot(test_dates, test_true, lw=1.2, color="#ff7f0e",
            label="Факт (тест)", zorder=3)

        # Прогноз (пунктирная красная)
        ax.plot(test_dates, test_pred, lw=1.2, color="#d62728", linestyle="--",
            label=f"Прогноз {model_name}", zorder=4)

        # Вертикальная линия разделения
        ax.axvline(test_dates[0], color="black", lw=1.5, linestyle=":",
               label="Начало теста", zorder=5)

        # Небольшая заливка тестовой области (опционально)
        ax.axvspan(test_dates[0], test_dates[-1], alpha=0.05, color="orange")

        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax.set_title(f"{model_name} | Горизонт h={horizon} | "
                 f"Прогноз лог-доходности OZON", fontsize=12)
        ax.set_ylabel("Лог-доходность")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.25)

        # ─── Нижний график: остатки (только на тесте) ───────────────────────────
        residuals = test_true - test_pred
        axes[1].bar(test_dates, residuals,
                color=["#d62728" if r < 0 else "#2ca02c" for r in residuals],
                width=1.0, alpha=0.7)
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_title("Остатки (факт − прогноз) на тестовой выборке")
        axes[1].set_ylabel("Остаток")
        axes[1].grid(alpha=0.25)

        plt.tight_layout()
        fname = f"fig_forecast_{model_name.lower()}_h{horizon}.png"
        return self._save(fname)

    def plot_wf_best_forecast(
        self,
        train_dates: pd.Index,
        train_true: np.ndarray,
        test_dates: pd.Index,
        test_true: np.ndarray,
        test_pred: np.ndarray,
        model_name: str,
        file_slug: str,
        horizon: int,
    ) -> Path:
        """
        Итоговый график факт vs прогноз для лучшей WF-модели.
        Файл: ``fig_<file_slug>_h<horizon>_forecast.png``.
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                                gridspec_kw={"height_ratios": [3, 1]})
        ax = axes[0]
        ax.plot(train_dates, train_true, lw=1.0, color="#1f77b4",
                label="Обучающая выборка", zorder=2)
        ax.plot(test_dates, test_true, lw=1.2, color="#ff7f0e",
                label="Факт (тест)", zorder=3)
        ax.plot(test_dates, test_pred, lw=1.2, color="#d62728", linestyle="--",
                label=f"Прогноз {model_name}", zorder=4)
        ax.axvline(test_dates[0], color="black", lw=1.5, linestyle=":",
                   label="Начало теста", zorder=5)
        ax.axvspan(test_dates[0], test_dates[-1], alpha=0.05, color="orange")
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax.set_title(
            f"{model_name} (лучшая по WF) | h={horizon} | лог-доходность OZON",
            fontsize=12,
        )
        ax.set_ylabel("Лог-доходность")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.25)

        residuals = test_true - test_pred
        axes[1].bar(
            test_dates, residuals,
            color=["#d62728" if r < 0 else "#2ca02c" for r in residuals],
            width=1.0, alpha=0.7,
        )
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_title("Остатки (факт − прогноз) на тестовой выборке")
        axes[1].set_ylabel("Остаток")
        axes[1].grid(alpha=0.25)

        plt.tight_layout()
        return self._save(f"fig_{file_slug}_h{horizon}_forecast.png")

    def plot_wf_best_residuals_diagnostics(
        self,
        preds_df: pd.DataFrame,
        model_name: str,
        file_slug: str,
        horizon: int,
    ) -> Optional[Path]:
        """
        Q-Q и ACF остатков для лучшей модели.
        Файл: ``fig_<file_slug>_h<horizon>_residuals.png``.
        """
        pred_col = f"pred_{model_name}"
        if pred_col not in preds_df.columns:
            logger.warning("[PLOT] Нет колонки %s", pred_col)
            return None
        residuals = (preds_df["y_true_reg"] - preds_df[pred_col]).dropna()
        if len(residuals) < 10:
            return None

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(preds_df.index, residuals, lw=0.7, color="steelblue")
        axes[0].axhline(0, color="red", lw=0.8)
        axes[0].set_title(f"Остатки {model_name} (h={horizon})")
        axes[0].set_ylabel("Ошибка")
        axes[0].grid(alpha=0.3)

        stats.probplot(residuals.values, plot=axes[1])
        axes[1].set_title(f"Q-Q: {model_name}")
        axes[1].grid(alpha=0.3)

        plot_acf(residuals.values, lags=20, ax=axes[2],
                 title=f"ACF остатков: {model_name}")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        return self._save(f"fig_{file_slug}_h{horizon}_residuals.png")

    # ─── Остатки ────────────────────────────────────────────────────────────

    def plot_residuals(
        self, preds_df: pd.DataFrame, model_name: str
    ) -> Optional[Path]:
        pred_col = f"pred_{model_name}"
        if pred_col not in preds_df.columns:
            logger.warning(f"[PLOT] Колонка {pred_col} не найдена")
            return None

        residuals = (preds_df["y_true_reg"] - preds_df[pred_col]).dropna()
        if len(residuals) < 10:
            return None

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # 1. Остатки во времени
        axes[0].plot(preds_df.index, residuals, lw=0.7, color="steelblue")
        axes[0].axhline(0, color="red", lw=0.8)
        axes[0].set_title(f"Остатки {model_name} во времени")
        axes[0].set_ylabel("Ошибка")
        axes[0].grid(alpha=0.3)

        # 2. Q-Q plot
        stats.probplot(residuals.values, plot=axes[1])
        axes[1].set_title(f"Q-Q plot: {model_name}")
        axes[1].grid(alpha=0.3)

        # 3. ACF остатков
        plot_acf(residuals.values, lags=20, ax=axes[2],
                 title=f"ACF остатков: {model_name}")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        return self._save(f"fig_residuals_{model_name.lower()}.png")

    # ─── Feature importance ──────────────────────────────────────────────────

    def plot_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        model_name: str,
        top_n: int = 20,
    ) -> Optional[Path]:
        try:
            imp = model.feature_importances_
        except AttributeError:
            try:
                imp = np.abs(model.coef_)
            except AttributeError:
                logger.warning(f"[PLOT] {model_name}: нет feature_importances_")
                return None

        indices = np.argsort(imp)[-top_n:]
        fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.3)))
        ax.barh([feature_names[i] for i in indices],
                imp[indices], color="steelblue")
        ax.set_title(f"Важность признаков: {model_name} (топ {top_n})")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        return self._save(f"fig_importance_{model_name.lower()}.png")

    # ─── Все графики разом ───────────────────────────────────────────────────

    def save_all(self, df: pd.DataFrame) -> List[Path]:
        saved = []
        for fn in [self.plot_price_volume, self.plot_returns,
                   self.plot_seasonality, self.plot_correlation]:
            try:
                p = fn(df)
                if p:
                    saved.append(p)
            except Exception as exc:
                logger.warning(f"[PLOT] {fn.__name__} failed: {exc}")
        return saved