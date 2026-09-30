"""src.RAG.schemas

Схемы данных, используемые при обработке файлов внутренней документации.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
from dataclasses import dataclass

from src.interfaces import (
    SourceProtocol, DocumentMetaProtocol, DocumentProtocol
)


@dataclass
class Source(SourceProtocol):
    """Файл-источник документа.

    Используется для отслеживания происхождения данных и дедупликации.
    """
    id: str
    path: Path
    type: str
    content_hash: str
    updated_at: datetime


@dataclass
class DocumentMeta(DocumentMetaProtocol):
    """Метаданные документа для RAG-пайплайна."""
    embedding_model: str
    source: Source


@dataclass
class Document(DocumentProtocol):
    """Реализация структуры прочитанного внутреннего документа
    для пакета src.RAG.
    """
    id: str
    chunk: str
    chunk_hash: str
    meta: DocumentMeta
    embedding: np.ndarray | None = None
