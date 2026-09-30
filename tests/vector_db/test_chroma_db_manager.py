"""tests.vector_db.test_chroma_db_manager

Тесты менеджера векторной БД ChromaDBManager.
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.config import Config, VectorDB
from src.vector_db.chroma_db_manager import ChromaDBManager

# Метаданные коллекции, передаваемые в get_or_create_collection.
# Фиксируют метрику HNSW для детерминированной конфигурации индекса.
COLLECTION_METADATA = {"hnsw:space": "cosine"}


def _build_doc(
    doc_id: str,
    chunk: str,
    embedding: list[float],
    model: str = "model",
    source_id: str = "src1",
    source_path: str = "/docs/doc.txt",
    source_type: str = "text/plain",
    content_hash: str = "hash",
    updated_at: str = "2024-01-01T00:00:00",
    embedding_is_1d: bool = False,
) -> MagicMock:
    """Создаёт мок документа по протоколу DocumentProtocol.

    По умолчанию эмбеддинг имеет форму (1, dim) (старый формат).
    При embedding_is_1d=True — форму (dim,), как при пакетной генерации.
    """
    emb = np.array(embedding) if embedding_is_1d else np.array([embedding])
    doc = MagicMock()
    doc.id = doc_id
    doc.chunk = chunk
    doc.embedding = emb
    doc.meta.embedding_model = model
    doc.meta.source.id = source_id
    doc.meta.source.path = source_path
    doc.meta.source.type = source_type
    doc.meta.source.content_hash = content_hash
    doc.meta.source.updated_at.isoformat.return_value = updated_at
    return doc


class TestChromaDBManager:
    """Тесты для ChromaDBManager."""

    @pytest.fixture
    def mock_config(self):
        """Фикстура мок-конфигурации."""
        config = MagicMock(spec=Config)
        config.VECTOR_DB = VectorDB.CHROMA
        return config

    @pytest.fixture
    def mock_collection(self):
        """Фикстура мок-коллекции Chroma."""
        return MagicMock(name="collection")

    @pytest.fixture
    def mock_client(self, mock_collection):
        """Фикстура мок-клиента Chroma."""
        client = MagicMock(name="client")
        client.get_or_create_collection.return_value = mock_collection
        return client

    @pytest.fixture
    def manager(self, mock_client, mock_config):
        """Фикстура ChromaDBManager с замоканным клиентом."""
        with patch(
            "src.vector_db.base.init_client",
            return_value=mock_client,
        ):
            manager = ChromaDBManager(mock_config)
            return manager

    @pytest.mark.asyncio
    async def test_create_collection(
        self, manager, mock_client, mock_collection
    ):
        """create_collection должен вызвать get_or_create_collection
        с именем по умолчанию и метаданными коллекции."""
        collection = await manager.create_collection()

        assert collection is mock_collection
        mock_client.get_or_create_collection.assert_called_once_with(
            name="Agent-Assistant",
            metadata=COLLECTION_METADATA,
        )

    @pytest.mark.asyncio
    async def test_create_collection_custom_name(
        self, manager, mock_client, mock_collection
    ):
        """create_collection должен передать кастомное имя и метаданные."""
        collection = await manager.create_collection(name="custom")

        assert collection is mock_collection
        mock_client.get_or_create_collection.assert_called_once_with(
            name="custom",
            metadata=COLLECTION_METADATA,
        )

    @pytest.mark.asyncio
    async def test_collection_property_lazy_init(
        self, manager, mock_client, mock_collection
    ):
        """Свойство collection должно создать коллекцию
        при первом обращении с метаданными."""
        col = await manager.collection

        assert col is mock_collection
        mock_client.get_or_create_collection.assert_called_once_with(
            name="Agent-Assistant",
            metadata=COLLECTION_METADATA,
        )

    @pytest.mark.asyncio
    async def test_collection_property_cached(
        self, manager, mock_client, mock_collection
    ):
        """Свойство collection должно кэшировать коллекцию."""
        col1 = await manager.collection
        col2 = await manager.collection

        assert col1 is col2
        mock_client.get_or_create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_single_document(self, manager, mock_collection):
        """add должен сохранить один документ через collection.upsert."""
        doc = _build_doc(
            doc_id="1",
            chunk="chunk1",
            embedding=[0.1, 0.2],
        )

        await manager.add(doc)

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args[1]
        assert call_kwargs["ids"] == ["id_1"]
        assert call_kwargs["documents"] == ["chunk1"]
        assert call_kwargs["embeddings"] == [[0.1, 0.2]]

    @pytest.mark.asyncio
    async def test_add_single_document_1d_embedding(
        self, manager, mock_collection
    ):
        """add корректно сохраняет документ с одномерным эмбеддингом.

        Проверяет совместимость с форматом эмбеддингов (dim,),
        возвращаемых пакетной генерацией generate_embeddings_batch.
        """
        doc = _build_doc(
            doc_id="1",
            chunk="chunk1",
            embedding=[0.1, 0.2],
            embedding_is_1d=True,
        )

        await manager.add(doc)

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args[1]
        assert call_kwargs["ids"] == ["id_1"]
        assert call_kwargs["documents"] == ["chunk1"]
        assert call_kwargs["embeddings"] == [[0.1, 0.2]]

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self, manager, mock_collection):
        """add должен сохранить список документов через collection.upsert."""
        doc1 = _build_doc(
            doc_id="1", chunk="chunk1", embedding=[0.1, 0.2]
        )
        doc2 = _build_doc(
            doc_id="2",
            chunk="chunk2",
            embedding=[0.3, 0.4],
            source_id="src2",
            content_hash="hash2",
            updated_at="2024-01-02T00:00:00",
        )

        await manager.add([doc1, doc2])

        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args[1]
        assert call_kwargs["ids"] == ["id_1", "id_2"]
        assert call_kwargs["documents"] == ["chunk1", "chunk2"]
        assert call_kwargs["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_add_passes_metadata(self, manager, mock_collection):
        """add должен передавать метаданные документа в upsert."""
        doc = _build_doc(
            doc_id="1",
            chunk="chunk1",
            embedding=[0.5, 0.6],
            model="embed-model",
            source_id="src1",
            source_path="/docs/doc1.txt",
            source_type="text/plain",
            content_hash="hash1",
            updated_at="2024-03-15T12:00:00",
        )

        await manager.add(doc)

        call_kwargs = mock_collection.upsert.call_args[1]
        assert call_kwargs["metadatas"] == [
            {
                "embedding_model": "embed-model",
                "source_id": "src1",
                "source_path": "/docs/doc1.txt",
                "source_type": "text/plain",
                "content_hash": "hash1",
                "updated_at": "2024-03-15T12:00:00",
            }
        ]

    @pytest.mark.asyncio
    async def test_add_passes_metadata_multiple(
        self, manager, mock_collection
    ):
        """add должен передавать метаданные для каждого документа."""
        doc1 = _build_doc(
            doc_id="1",
            chunk="chunk1",
            embedding=[0.1, 0.2],
            source_id="src1",
            content_hash="hash1",
            updated_at="2024-01-01T00:00:00",
        )
        doc2 = _build_doc(
            doc_id="2",
            chunk="chunk2",
            embedding=[0.3, 0.4],
            source_id="src2",
            content_hash="hash2",
            updated_at="2024-01-02T00:00:00",
        )

        await manager.add([doc1, doc2])

        call_kwargs = mock_collection.upsert.call_args[1]
        assert call_kwargs["metadatas"] == [
            {
                "embedding_model": "model",
                "source_id": "src1",
                "source_path": "/docs/doc.txt",
                "source_type": "text/plain",
                "content_hash": "hash1",
                "updated_at": "2024-01-01T00:00:00",
            },
            {
                "embedding_model": "model",
                "source_id": "src2",
                "source_path": "/docs/doc.txt",
                "source_type": "text/plain",
                "content_hash": "hash2",
                "updated_at": "2024-01-02T00:00:00",
            },
        ]

    @pytest.mark.asyncio
    async def test_delete_by_source(self, manager, mock_collection):
        """delete_by_source удаляет записи источника через collection.delete."""
        await manager.delete_by_source("src1")

        mock_collection.delete.assert_called_once_with(
            where={"source_id": "src1"}
        )
