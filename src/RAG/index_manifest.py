"""src.RAG.index_manifest

Работа с манифестом индексации внутренней документации.

Манифест — это персистентный JSON-файл, хранящий состояние уже
проиндексированных файлов-источников. Для каждого источника сохраняется
его content_hash, дата последнего изменения и source_id.

Назначение манифеста:
  - инкрементальная индексация: при повторных запусках обрабатываются
    только изменённые и новые файлы, а неизменённые пропускаются;
  - устранение дублей записей в векторной БД за счёт отслеживания
    состояния источников.

Модуль предоставляет функции для загрузки/сохранения манифеста,
записи состояния источника и проверки факта изменения файла.
"""

import json
from pathlib import Path
from typing import Any

from src.config import Config
from src.logger import logger
from src.RAG.schemas import Source


def _manifest_path(config: Config) -> Path:
    """Возвращает путь к файлу манифеста индексации."""
    return config.INDEX_MANIFEST_PATH


def _ensure_dir(path: Path) -> None:
    """Создаёт родительскую директорию файла, если её нет."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _source_key(path: Path) -> str:
    """Возвращает согласованный строковый ключ для пути источника.

    Использует POSIX-представление пути (прямые слэши), чтобы ключи
    манифеста были единообразны вне зависимости от ОС (Windows/Linux),
    а также совпадали с путями, ожидаемыми в других сервисах.

    Args:
        path: Путь файла-источника.

    Returns:
        str: Строковый ключ пути с прямыми слэшами.
    """
    return path.as_posix()


def load_manifest(config: Config) -> dict[str, dict[str, Any]]:
    """Загружает манифест индексации из файла.

    Args:
        config: Конфигурация приложения с путём к манифесту.

    Returns:
        dict[str, dict[str, Any]]: Словарь вида
        {путь_файла: {content_hash, updated_at, source_id}}.
        Пустой словарь, если манифест отсутствует или повреждён.
    """
    manifest_path = _manifest_path(config)
    if not manifest_path.exists():
        return {}

    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning('манифест индексации имеет неверный формат')
            return {}
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f'не удалось загрузить манифест индексации: {e}')
        return {}


def save_manifest(
    config: Config,
    manifest: dict[str, dict[str, Any]]
) -> None:
    """Сохраняет манифест индексации в файл.

    Args:
        config: Конфигурация приложения с путём к манифесту.
        manifest: Словарь состояния индексации
        {путь_файла: {content_hash, updated_at, source_id}}.
    """
    manifest_path = _manifest_path(config)
    try:
        _ensure_dir(manifest_path)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.error(f'не удалось сохранить манифест индексации: {e}')


def record_source(
    config: Config,
    manifest: dict[str, dict[str, Any]],
    source: Source
) -> None:
    """Записывает состояние источника в манифест.

    Args:
        config: Конфигурация приложения.
        manifest: Словарь состояния индексации (мутируется на месте).
        source: Источник (файл), который был проиндексирован.
    """
    path = _source_key(source.path)
    manifest[path] = {
        'content_hash': source.content_hash,
        'updated_at': source.updated_at.isoformat(),
        'source_id': source.id,
    }


def is_source_changed(config: Config, source: Source) -> bool:
    """Определяет, изменился ли источник относительно манифеста.

    Args:
        config: Конфигурация приложения.
        source: Источник (файл) для проверки.

    Returns:
        bool: True, если файл новый или его содержимое изменилось
        относительно сохранённого состояния.
    """
    manifest = load_manifest(config)
    path = _source_key(source.path)
    record = manifest.get(path)
    if record is None:
        return True
    return record.get('content_hash') != source.content_hash
