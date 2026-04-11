"""
Скачать модель тональности с Hugging Face в локальный каталог (без запуска пайплайна).

Дальше в config.py задайте:
  NEWS_MODEL = r\"полный\\путь\\к\\каталогу\"

Пример:
  python scripts/download_news_model.py --out data/models/rubert-tiny-sentiment

Токен (если нужен): переменная HF_TOKEN или аргумент --token
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Скачать NEWS_MODEL с Hugging Face Hub")
    p.add_argument(
        "--repo",
        default="seara/rubert-tiny2-russian-sentiment",
        help="Идентификатор репозитория на Hub",
    )
    p.add_argument(
        "--revision",
        default="main",
        help="Ветка или тег (например main)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "data" / "models" / "rubert-tiny-sentiment-balanced",
        help="Каталог для файлов модели",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Токен HF (иначе берётся HF_TOKEN из окружения)",
    )
    args = p.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("Установите huggingface_hub: pip install huggingface_hub") from e

    args.out.mkdir(parents=True, exist_ok=True)
    token_arg = token if token else None
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=str(args.out),
        token=token_arg,
        revision=args.revision,
    )
    resolved = Path(path).resolve() if path else args.out.resolve()
    print("Модель сохранена:", resolved)
    print("В config.py укажите абсолютный путь, например:")
    print(f'  "NEWS_MODEL": r"{resolved}",')


if __name__ == "__main__":
    main()
