"""tests.vector_db.test_client

Тесты модуля инициализации клиента векторной БД.
"""

from unittest.mock import patch, MagicMock

import pytest

from src.config import Config, VectorDB
from src.vector_db.client import init_client


class TestInitClient:
    """Тесты для init_client."""

    @patch("src.vector_db.client.chromadb")
    def test_init_client_chroma(self, mock_chromadb):
        """Должен вызвать chromadb.Client() для VectorDB.CHROMA."""
        mock_client = MagicMock()
        mock_chromadb.Client.return_value = mock_client

        config = MagicMock(spec=Config)
        config.VECTOR_DB = VectorDB.CHROMA

        result = init_client(config)

        assert result is mock_client
        mock_chromadb.Client.assert_called_once_with()

    def test_init_client_unsupported_db_raises_value_error(self):
        """Должен возбудить ValueError для неподдерживаемой БД."""
        config = MagicMock(spec=Config)
        config.VECTOR_DB = "unsupported"

        with pytest.raises(ValueError, match="Неподдерживаемая векторная БД"):
            init_client(config)
