"""src.RAG.schemas

Схемы данных, используемые при обработке файлов внутренней документации.
"""

import numpy as np

from dataclasses import dataclass, field

from src.interfaces import Document as ReadedDocument


@dataclass
class Document(ReadedDocument):
    """Реализация структуры прочитанного внутреннего документа - ReadedDocument
    для пакета src.RAG.
    """
    chunks: list[str] = field(default_factory=list)
    embeddings: np.ndarray | None = None
