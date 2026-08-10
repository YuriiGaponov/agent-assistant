"""tests.vector_db.test_init_vector_db_manager

Тесты фабричной функции get_vectorDB_manager.
"""

from unittest.mock import patch, MagicMock

import pytest

from src.config import Config, VectorDB
from src.vector_db import get_vectorDB_manager
from src.vector_db.chroma_db_manager import ChromaDBManager


class TestGetVectorDBManager:
    """Тесты для get_vectorDB_manager."""

    @patch("src.vector_db.base.init_client")
    def test_returns_chroma_db_manager(self, mock_init_client):
        """Должен вернуть экземпляр ChromaDBManager для VectorDB.CHROMA."""
        config = MagicMock(spec=Config)
        config.VECTOR_DB = VectorDB.CHROMA

        manager = get_vectorDB_manager(config)

        assert isinstance(manager, ChromaDBManager)
        mock_init_client.assert_called_once_with(config)

    @patch("src.vector_db.base.init_client")
    def test_unsupported_db_raises_value_error(self, mock_init_client):
        """Должен возбудить ValueError для неподдерживаемой БД."""
        config = MagicMock(spec=Config)
        config.VECTOR_DB = "unsupported"

        with pytest.raises(ValueError, match="Неподдерживаемая векторная БД"):
            get_vectorDB_manager(config)

        mock_init_client.assert_not_called()
