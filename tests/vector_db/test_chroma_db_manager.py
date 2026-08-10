"""tests.vector_db.test_chroma_db_manager

Тесты менеджера векторной БД ChromaDBManager.
"""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.config import Config, VectorDB
from src.vector_db.chroma_db_manager import ChromaDBManager


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
        с именем по умолчанию."""
        collection = await manager.create_collection()

        assert collection is mock_collection
        mock_client.get_or_create_collection.assert_called_once_with(
            "Agent-Assistant"
        )

    @pytest.mark.asyncio
    async def test_create_collection_custom_name(
        self, manager, mock_client, mock_collection
    ):
        """create_collection должен передать кастомное имя."""
        collection = await manager.create_collection(name="custom")

        assert collection is mock_collection
        mock_client.get_or_create_collection.assert_called_once_with("custom")

    @pytest.mark.asyncio
    async def test_collection_property_lazy_init(
        self, manager, mock_client, mock_collection
    ):
        """Свойство collection должно создать коллекцию
        при первом обращении."""
        col = await manager.collection

        assert col is mock_collection
        mock_client.get_or_create_collection.assert_called_once_with(
            "Agent-Assistant"
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
        """add должен добавить один документ с чанками и эмбеддингами."""
        doc = MagicMock()
        doc.chunks = ["chunk1", "chunk2"]
        doc.embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])

        await manager.add(doc)

        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args[1]
        assert call_kwargs["ids"] == ["id_1", "id_2"]
        assert call_kwargs["documents"] == ["chunk1", "chunk2"]
        assert call_kwargs["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self, manager, mock_collection):
        """add должен добавить список документов."""
        doc1 = MagicMock()
        doc1.chunks = ["chunk1"]
        doc1.embeddings = np.array([[0.1, 0.2]])

        doc2 = MagicMock()
        doc2.chunks = ["chunk2"]
        doc2.embeddings = np.array([[0.3, 0.4]])

        await manager.add([doc1, doc2])

        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args[1]
        assert call_kwargs["ids"] == ["id_1", "id_2"]
        assert call_kwargs["documents"] == ["chunk1", "chunk2"]
        assert call_kwargs["embeddings"] == [[0.1, 0.2], [0.3, 0.4]]

    @pytest.mark.asyncio
    async def test_add_document_no_embeddings(self, manager, mock_collection):
        """add должен пропустить документ без эмбеддингов."""
        doc = MagicMock()
        doc.chunks = ["chunk1"]
        doc.embeddings = None

        await manager.add(doc)

        mock_collection.add.assert_called_once_with(
            ids=[],
            embeddings=[],
            documents=[],
        )
