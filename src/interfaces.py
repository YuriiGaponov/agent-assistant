"""src.interfaces

Абстрактные классы и протоколы, определяющие контракты
различных модулей и пакетов, а также внешних систем проекта Agent‑Assistant.
"""

from typing import Protocol, Sequence, Literal

import numpy as np


# ======= Протоколы взаимодействия с ИИ‑агентами =======
class UserQueryProtocol(Protocol):
    """Протокол запроса пользователя, передаваемого в агент для выполнения."""
    query: str


class AgentProtocol(Protocol):
    """Протокол менеджера взаимодействия с LLM."""
    async def health_check(self) -> bool:
        """Проверка доступности LLM. Возвращает True, если LLM доступна
        должна отвечать за разумное время.
        """
        pass

    async def chat(self, query: UserQueryProtocol) -> str:
        """Взаимодействие с LLM в виде диалога.
        Принимает объект запроса пользователя,
        возвращает строковый ответ.
        """
        pass


# ======= Протоколы обработки внутренней документации =======
class Document(Protocol):
    """Структура данных прочитанного документа внутренней документации,
    содержащая поля, которые необходимо внести в векторную БД.
    """
    chunks: list[str]
    embeddings: np.ndarray | None = None


# ======= Протоколы взаимодействия с векторной базой данных =======
class EmbeddingModelProtocol(Protocol):
    """Протокол модели генерации эмбеддингов"""
    def encode(
        self, chunks: list[str],
        convert_to_numpy: bool
    ) -> np.ndarray:
        pass


class VectorSearchQueryProtocol(Protocol):
    """Протокол объекта запроса для семантического поиска в векторной БД"""
    query: str


class VectorSearchResultProtocol(Protocol):
    """Протокол объекта результата семантического поиска в векторной БД,
    возвращаемого менеджером векторной БД.
    """
    documents: Sequence[str]


class VectorDBManagerInterface(Protocol):
    """Протокол менеджера векторной БД: определяет ожидаемый интерфейс для
        - создания векторной коллекции,
        - добавления документов в векторное хранилище,
        - семантического поиска по вектору запроса.
    """
    async def create_collection(self, name: str = 'Agent-Assistant'):
        """Получает существующую или создает новую коллекцию векторной БД
        с заданным именем name.
        """
        pass

    async def add(self, data: Document | list[Document]) -> None:
        """Принимает один  объект или список структуры данных Document,
        определенной в пакете RAG, содержащей данные, которые должны быть
        добавлены в векторную БД.
        """
        pass

    async def search(self, embeddings: np.ndarray):
        """Принимает преобразованный в эмбеддинги поисковый запрос
        для семантического поиска в векторной БД.
        """
        pass


# ======= Протоколы проверки работоспособности агента =======
class HealthStatusProtocol(Protocol):
    """Протокол состояния работоспособности систем Agent-Assistant"""
    # Общий статус
    status: Literal["healthy", "unhealthy"]
    # Доступность хранилища внутренней документации
    doc_store_available: bool
    # Доступность главной LLM.
    main_llm_available: bool
