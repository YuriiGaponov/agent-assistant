"""src.agent.tools

Фабрика инструментов для AI‑агента в проекте Agent‑Assistant.

Предоставляет:
    - Функцию make_tools, которая создаёт список инструментов LangChain
      на основе переданных зависимостей (модель эмбеддингов, векторная БД).
    - Инструмент semantic_search, который агент может вызывать для поиска
      релевантных фрагментов в векторной базе.

Масштабируемость: список инструментов легко расширяется — достаточно
добавить новый инструмент в make_tools, и граф агента автоматически
передаст его модели через привязку инструментов.
"""

from langchain_core.tools import tool
from pydantic import Field

from src.interfaces import (
    EmbeddingModelProtocol, VectorDBManagerInterface
)
from src.RAG import semantic_search as _semantic_search


def make_tools(
        embedding_model: EmbeddingModelProtocol,
        vector_db: VectorDBManagerInterface
):
    """Фабрика инструментов агента.

    Создаёт и возвращает список инструментов, которые агент может использовать.
    На текущий момент содержит только инструмент semantic_search.

    Args:
        embedding_model: Модель эмбеддингов (реализует EmbeddingModelProtocol).
        vector_db: Менеджер векторной БД (реализует VectorDBManagerInterface).

    Returns:
        List[Any]: Список инструментов LangChain для передачи в агент.
    """
    @tool
    async def semantic_search(
        query: str = Field(
            description="Текст запроса пользователя для семантического поиска."
        )
    ) -> str:
        """
        Семантический поиск по базе знаний.

        Агент использует этот инструмент, чтобы найти релевантные фрагменты
        документов на основе смысловой близости к запросу пользователя.

        Логика работы:
            1. На вход получает текстовый запрос.
            2. Выполняет поиск по векторной базе внутренней документации.
            3. Возвращает найденные фрагменты текстом (разделённые
               переносами строки) для формирования обобщённого ответа.

        Возвращаемое значение:
            Строка с найденными фрагментами документов, разделёнными
            пустой строкой.
        """
        result = await _semantic_search(query, embedding_model, vector_db)
        documents = result.get('documents', [[]])
        fragments = list(documents[0]) if documents else []
        return '\n\n'.join(fragments) if fragments else 'Фрагменты не найдены.'

    return [semantic_search]
