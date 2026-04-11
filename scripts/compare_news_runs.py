"""
Сравнение двух прогонов pipeline (с/без новостей) по сохранённым метрикам.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CFG
from pipeline import run_experiment


def summarize(tag: str, ml: dict, *, metric: str) -> list[dict]:
    """Лучшая регрессия по той же метрике, что и в pipeline (BEST_MODEL_METRIC)."""
    out: list[dict] = []
    minimize = metric != "R2"
    for h, df in ml["wf_reg"].items():
        if (
            isinstance(df, pd.DataFrame)
            and not df.empty
            and "model" in df.columns
            and metric in df.columns
        ):
            g = df.groupby("model", as_index=False)[metric].mean()
            g = g.sort_values(metric, ascending=minimize)
            best = g.iloc[0]
            out.append(
                {
                    "tag": tag,
                    "h": int(h),
                    "best_reg_model": best["model"],
                    f"best_reg_{metric.lower()}": float(best[metric]),
                }
            )
    for h, df in ml["wf_clf"].items():
        if isinstance(df, pd.DataFrame) and not df.empty and "model" in df.columns and "Accuracy" in df.columns:
            g = (
                df.groupby("model", as_index=False)["Accuracy"]
                .mean()
                .sort_values("Accuracy", ascending=False)
            )
            best = g.iloc[0]
            rec = next((r for r in out if r["h"] == int(h) and r["tag"] == tag), None)
            if rec is None:
                rec = {"tag": tag, "h": int(h)}
                out.append(rec)
            rec["best_clf_model"] = best["model"]
            rec["best_clf_acc"] = float(best["Accuracy"])
    return out


def main() -> None:
    cfg_with = dict(CFG)
    cfg_with["USE_NEWS_SENTIMENT"] = True
    res_with = run_experiment(cfg_with, "OZON_combined.csv", "results_with_news_cmp")

    cfg_no = dict(CFG)
    cfg_no["USE_NEWS_SENTIMENT"] = False
    res_no = run_experiment(cfg_no, "OZON_combined.csv", "results_no_news_cmp")

    m = str(CFG.get("BEST_MODEL_METRIC", "RMSE"))
    rows = summarize("with_news", res_with["ml"], metric=m) + summarize(
        "no_news", res_no["ml"], metric=m
    )
    df = pd.DataFrame(rows).sort_values(["h", "tag"])
    out = Path("results_compare_summary.csv")
    out.write_text(df.to_csv(index=False), encoding="utf-8")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

