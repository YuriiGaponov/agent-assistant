"""tests.services.RAG.test_index_manifest

Тесты модуля работы с манифестом индексации.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

from src.RAG.index_manifest import (
    load_manifest,
    save_manifest,
    record_source,
    is_source_changed,
)
from src.RAG.schemas import Source


def _make_source(
    path: str = "/docs/doc.txt",
    content_hash: str = "hash1",
) -> Source:
    """Создаёт объект Source для тестов."""
    from datetime import datetime

    return Source(
        id=content_hash,
        path=Path(path),
        type="text/plain",
        content_hash=content_hash,
        updated_at=datetime(2024, 1, 1),
    )


def test_load_manifest_missing_file_returns_empty(
    tmp_path: Path,
) -> None:
    """load_manifest возвращает пустой словарь при отсутствии файла."""
    config = MagicMock()
    config.INDEX_MANIFEST_PATH = tmp_path / "manifest.json"

    manifest = load_manifest(config)

    assert manifest == {}


def test_save_and_load_manifest_roundtrip(
    tmp_path: Path,
) -> None:
    """save_manifest сохраняет, а load_manifest загружает манифест."""
    config = MagicMock()
    manifest_path = tmp_path / "manifest.json"
    config.INDEX_MANIFEST_PATH = manifest_path

    manifest_data = {
        "/docs/doc.txt": {
            "content_hash": "hash1",
            "updated_at": "2024-01-01T00:00:00",
            "source_id": "hash1",
        }
    }
    save_manifest(config, manifest_data)

    loaded = load_manifest(config)
    assert loaded == manifest_data


def test_save_manifest_creates_parent_dir(
    tmp_path: Path,
) -> None:
    """save_manifest создаёт родительские директории файла."""
    config = MagicMock()
    manifest_path = tmp_path / "sub" / "dir" / "manifest.json"
    config.INDEX_MANIFEST_PATH = manifest_path

    save_manifest(config, {})

    assert manifest_path.exists()


def test_record_source_adds_entry() -> None:
    """record_source добавляет запись источника в манифест."""
    config = MagicMock()
    source = _make_source(path="/docs/a.txt", content_hash="hash_a")
    manifest: dict = {}

    record_source(config, manifest, source)

    assert manifest == {
        "/docs/a.txt": {
            "content_hash": "hash_a",
            "updated_at": "2024-01-01T00:00:00",
            "source_id": "hash_a",
        }
    }


def test_is_source_changed_returns_true_for_new_source(
    tmp_path: Path,
) -> None:
    """Новый источник (отсутствует в манифесте) считается изменённым."""
    config = MagicMock()
    config.INDEX_MANIFEST_PATH = tmp_path / "manifest.json"
    source = _make_source()

    assert is_source_changed(config, source) is True


def test_is_source_changed_false_when_unchanged(
    tmp_path: Path,
) -> None:
    """Источник с тем же content_hash не считается изменённым."""
    config = MagicMock()
    manifest_path = tmp_path / "manifest.json"
    config.INDEX_MANIFEST_PATH = manifest_path
    manifest = {
        "/docs/doc.txt": {
            "content_hash": "hash1",
            "updated_at": "2024-01-01T00:00:00",
            "source_id": "hash1",
        }
    }
    save_manifest(config, manifest)

    source = _make_source(path="/docs/doc.txt", content_hash="hash1")

    assert is_source_changed(config, source) is False


def test_is_source_changed_true_when_content_changed(
    tmp_path: Path,
) -> None:
    """Источник с изменённым content_hash считается изменённым."""
    config = MagicMock()
    manifest_path = tmp_path / "manifest.json"
    config.INDEX_MANIFEST_PATH = manifest_path
    manifest = {
        "/docs/doc.txt": {
            "content_hash": "hash_old",
            "updated_at": "2024-01-01T00:00:00",
            "source_id": "hash_old",
        }
    }
    save_manifest(config, manifest)

    source = _make_source(path="/docs/doc.txt", content_hash="hash_new")

    assert is_source_changed(config, source) is True
