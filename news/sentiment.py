"""
Тональность новостей: HuggingFace ``AutoModelForSequenceClassification``.

Поддерживаются:

- **Модель с Hub**, уже обученная под классификацию (по умолчанию
  ``seara/rubert-tiny2-russian-sentiment``; можно ``cointegrated/rubert-tiny-sentiment-balanced``).
- **Локальный каталог** после дообучения своей головы на базе вроде
  ``cointegrated/rubert-tiny2``: обучить классификационную голову, затем
  ``model.save_pretrained(path)``, ``tokenizer.save_pretrained(path)`` —
  в ``config.json`` должны быть ``id2label`` / ``label2id`` и веса головы
  в чекпоинте (без «missing» для ``classifier``). В ``config.py`` задайте
  ``NEWS_MODEL`` = этот путь.

Базовый encoder-only чекпоинт с Hub без дообучения (только MLM) **не**
подходит: в нём нет обученной головы ``ForSequenceClassification``.

Скаляр тональности: ``proba @ w`` (веса −1/0/+1 по меткам в ``id2label``). Для
multi-label в конфиге — ``sigmoid(logits)``, иначе ``softmax(logits)`` (стандартный CE).
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Совпадает с дефолтом в config.CFG["NEWS_MODEL"] (альтернатива: cointegrated/rubert-tiny-sentiment-balanced)
DEFAULT_NEWS_MODEL = "seara/rubert-tiny2-russian-sentiment"


def _darwin_torch_mitigations() -> None:
    """
    До ``import torch``: уменьшает риск segmentation fault на macOS (часто OpenMP/BLAS + PyTorch 3.13).
    """
    if platform.system() != "Darwin":
        return
    # Дублирующиеся libomp / гонки потоков — типичный источник SIGSEGV на Apple Silicon / Intel Mac
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    for key, val in (
        ("OMP_NUM_THREADS", "1"),
        ("MKL_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
        ("VECLIB_MAXIMUM_THREADS", "1"),
        ("NUMEXPR_NUM_THREADS", "1"),
    ):
        os.environ.setdefault(key, val)


def _resolve_hf_token(cfg_token: str | None) -> str | None:
    """
    Токен для Hub: сначала ``NEWS_HF_TOKEN`` из конфига, иначе ``HF_TOKEN`` /
    ``HUGGING_FACE_HUB_TOKEN`` из окружения (так надёжнее, чем надеяться на неявный pick-up).
    """
    if cfg_token:
        s = str(cfg_token).strip()
        if s:
            logger.info("[NEWS-SENT] HF token: из конфига NEWS_HF_TOKEN")
            return s
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = os.environ.get(name, "").strip()
        if v:
            logger.info("[NEWS-SENT] HF token: из переменной окружения %s", name)
            return v
    logger.info(
        "[NEWS-SENT] HF token не задан — для публичных моделей это нормально"
    )
    return None


def _hub_load_failure_message(model_ref: str, original: Exception) -> str:
    """Подсказки при 401/404 / сетевых ошибках / неверном токене."""
    msg = str(original).lower()
    lines = [
        f"Не удалось загрузить модель «{model_ref}» с Hugging Face Hub.",
        f"Техническое сообщение: {original!r}",
        "",
        "Что сделать по шагам:",
        "1) Проверьте доступ к https://huggingface.co в браузере (VPN, файрвол, корпоративная сеть).",
        "2) Если в окружении задан устаревший HF_TOKEN — выполните `huggingface-cli logout` "
        "или удалите переменную HF_TOKEN и перезапустите терминал (для публичной модели токен не нужен).",
        "3) Для доступа к приватным репозиториям: `huggingface-cli login` или "
        "NEWS_HF_TOKEN / HF_TOKEN с актуальным токеном с https://huggingface.co/settings/tokens",
        "4) Офлайн: скачайте модель на машину с интернетом и укажите путь к каталогу с config.json:",
        "   python scripts/download_news_model.py --out data/models/sentiment",
        "   затем NEWS_MODEL = абсолютный путь к этой папке.",
        "5) Временно отключите тональность: USE_NEWS_SENTIMENT = False в config.py.",
        "",
        "При 404 часто помогает: сбросить HF_ENDPOINT (зеркала вроде hf-mirror отдают 404 для части файлов). "
        "В Linux/macOS: unset HF_ENDPOINT; в Windows CMD: set HF_ENDPOINT=.",
        "Попробуйте сменить NEWS_MODEL на другую публичную модель: "
        "seara/rubert-tiny2-russian-sentiment или cointegrated/rubert-tiny-sentiment-balanced.",
    ]
    if "401" in msg or "403" in msg or "unauthorized" in msg:
        lines.insert(
            2,
            "(Похоже на 401: неверный токен, истёкшая сессия или репозиторий недоступен.)",
        )
    if "404" in msg or "not found" in msg:
        lines.insert(
            2,
            "(Похоже на 404: неверное имя репозитория, зеркало HF_ENDPOINT или ревизия ветки.)",
        )
    return "\n".join(lines)


def _is_local_checkpoint_dir(ref: str) -> bool:
    p = Path(ref).expanduser()
    return p.is_dir() and (p / "config.json").is_file()


def _resolve_news_model_ref(model_name: str) -> str:
    """
    Hub id или абсолютный путь к каталогу с ``config.json`` (свой чекпоинт).
    """
    p = Path(model_name).expanduser()
    if p.is_dir() and (p / "config.json").is_file():
        resolved = str(p.resolve())
        logger.info("[NEWS-SENT] Локальный чекпоинт: %s", resolved)
        return resolved
    return model_name


def _missing_keys_from_loading_info(info: Any) -> List[str]:
    """Совместимость с разными версиями transformers (tuple vs dict)."""
    if isinstance(info, dict):
        return list(info.get("missing_keys", []))
    if isinstance(info, (list, tuple)) and len(info) >= 1:
        return list(info[0])
    return []


def _label_to_score(label: str) -> float:
    s = str(label).lower()
    if any(x in s for x in ("neg", "негат")):
        return -1.0
    if any(x in s for x in ("pos", "позит")):
        return 1.0
    return 0.0


def _class_weight_vector(id2label: Dict[int, str], num_labels: int) -> np.ndarray:
    """
    Веса для скаляра тональности: как ``proba.dot([-1, 0, 1])`` в документации
    ``cointegrated/rubert-tiny-sentiment-balanced``, с опорой на ``id2label[i]`` (neg/neu/pos).
    """
    w = np.zeros(int(num_labels), dtype=np.float64)
    for i in range(int(num_labels)):
        w[i] = _label_to_score(id2label.get(i, ""))
    return w


def predict_sentiment_batch(
    texts: Sequence[str],
    *,
    model_name: str,
    batch_size: int = 8,
    device: str | None = None,
    max_length: int = 512,
    hf_token: str | None = None,
    hf_revision: str | None = None,
) -> np.ndarray:
    texts_list = list(texts)
    n = len(texts_list)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # До import torch: токенизатор и BLAS (важно на macOS)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _darwin_torch_mitigations()

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except (ImportError, OSError) as e:
        raise RuntimeError(
            "Не удалось загрузить torch/transformers для RuBERT. "
            "На Windows при WinError 1114 установите VC++ Redistributable (x64) "
            "или переустановите PyTorch (CPU): https://pytorch.org/get-started/locally/ "
            "Либо добавьте в CSV новостей готовую колонку «sentiment» и инференс будет пропущен."
        ) from e

    if platform.system() == "Darwin":
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        except Exception as exc:
            logger.warning("[NEWS-SENT] torch.set_num_threads: %s", exc)
        logger.info(
            "[NEWS-SENT] macOS: потоки PyTorch=1 (снижает риск segfault с transformers)"
        )

    device_t = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # MPS на части сборок PyTorch 3.13 падает в SIGSEGV на BERT — по умолчанию CPU надёжнее
    if device_t == "mps":
        logger.warning(
            "[NEWS-SENT] NEWS_DEVICE=mps: на macOS возможен segfault; переключаюсь на cpu"
        )
        device_t = "cpu"

    model_ref = _resolve_news_model_ref(model_name)
    tok = _resolve_hf_token(hf_token)
    hub_kw: Dict[str, Any] = {}
    if tok:
        hub_kw["token"] = tok
    if not _is_local_checkpoint_dir(model_ref):
        rev = (hf_revision or "main").strip() or "main"
        hub_kw["revision"] = rev
        logger.info("[NEWS-SENT] Hugging Face revision=%s", rev)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_ref, **hub_kw)
        raw = AutoModelForSequenceClassification.from_pretrained(
            model_ref,
            output_loading_info=True,
            **hub_kw,
        )
    except Exception as e:
        if _is_local_checkpoint_dir(model_name) or _is_local_checkpoint_dir(model_ref):
            raise
        raise RuntimeError(_hub_load_failure_message(model_ref, e)) from e
    if isinstance(raw, tuple) and len(raw) == 2:
        model, loading_info = raw[0], raw[1]
    else:
        model = raw  # type: ignore[assignment]
        loading_info = None

    missing_keys = _missing_keys_from_loading_info(loading_info)

    # Базовый BERT (MLM): в чекпоинте cls.*, а голова classifier.* не обучена — веса «missing»
    clf_missing = [
        k
        for k in missing_keys
        if any(x in k for x in ("classifier", "pre_classifier", "score"))
    ]
    if clf_missing:
        raise RuntimeError(
            f"В чекпоинте «{model_name}» нет весов классификационной головы: {clf_missing}. "
            "Базовые модели вроде cointegrated/rubert-tiny2 с Hub в таком виде не подходят: "
            "нужно самостоятельно дообучить классификационную голову, сохранить через "
            "model.save_pretrained(path) и tokenizer.save_pretrained(path) в формате "
            "AutoModelForSequenceClassification с нужными id2label/label2id, затем указать "
            "NEWS_MODEL=path к этому каталогу. Либо укажите публичную sentiment-модель с Hub "
            f"(например {DEFAULT_NEWS_MODEL})."
        )

    model.to(device_t)
    model.eval()

    id2label: Dict[int, str] = {
        int(k): str(v) for k, v in model.config.id2label.items()
    }
    num_labels = int(getattr(model.config, "num_labels", len(id2label)))
    class_w = _class_weight_vector(id2label, num_labels)
    logger.info("[NEWS-SENT] %s, device=%s", model_ref, device_t)

    out_list: List[float] = []
    with torch.inference_mode():
        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device_t) for k, v in enc.items()}
            logits = model(**enc).logits
            # multi_label → sigmoid (как в карточке cointegrated/*); иначе softmax (обычный CE)
            pt = getattr(model.config, "problem_type", None) or ""
            if pt == "multi_label_classification":
                proba = torch.sigmoid(logits).cpu().numpy()
            else:
                proba = torch.softmax(logits, dim=-1).cpu().numpy()
            if proba.ndim == 1:
                proba = proba.reshape(1, -1)
            if class_w.size != proba.shape[1]:
                raise RuntimeError(
                    f"Размерность proba {proba.shape[1]} не совпадает с id2label ({len(id2label)}). "
                    "Проверьте модель."
                )
            scores = proba @ class_w
            out_list.extend(float(x) for x in scores)

    return np.asarray(out_list, dtype=np.float64)


def ensure_sentiment_column(
    df: pd.DataFrame,
    text_col: str,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    if "sentiment" in df.columns:
        logger.info("[NEWS-SENT] sentiment уже есть, пропуск инференса")
        return df

    out = df.copy()
    model_name = str(cfg.get("NEWS_MODEL", DEFAULT_NEWS_MODEL))
    batch_size = int(cfg.get("NEWS_BATCH_SIZE", 8))
    device = cfg.get("NEWS_DEVICE")
    max_length = int(cfg.get("NEWS_MAX_LENGTH", 512))
    raw_tok = cfg.get("NEWS_HF_TOKEN")
    hf_explicit = str(raw_tok).strip() if raw_tok else None
    raw_rev = cfg.get("NEWS_HF_REVISION")
    hf_revision = str(raw_rev).strip() if raw_rev else None

    texts = out[text_col].astype(str).tolist()
    scores = predict_sentiment_batch(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        device=device if device is not None else None,
        max_length=max_length,
        hf_token=hf_explicit,
        hf_revision=hf_revision,
    )
    out["sentiment"] = scores
    return out
