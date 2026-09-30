"""src.agent.__init__

Пакет реализации агентов в проекте Agent‑Assistant.

Предоставляет:
    - Фабрику init_agent для создания агента в зависимости от выбранного
      источника LLM (Ollama, Mistral, OpenRouter и т.д.).
    - Изоляцию логики выбора и инициализации агента от остального кода,
      что упрощает расширение списка поддерживаемых провайдеров и тестирование.
    - Единый контракт через интерфейс AgentProtocol, гарантирующий наличие
      стандартных методов (health_check, chat) у любого агента.

Пакет отвечает за оркестрацию (построение графа LangGraph и взаимодействие
с LLM через интерфейс LLMManagerInterface). Вся низкоуровневая логика
работы с моделью вынесена в отдельный пакет src.llm.

Основные сценарии использования:
    1. Инициализация агента на старте приложения (в startup‑хендлере)
       и сохранение в app.state для использования во всех сервисах.
    2. Внедрение фабрики (или уже созданного агента) через зависимости
       в роутеры и сервисы, где требуется выполнение агентских задач.
    3. Тестирование: фабрика позволяет подменять реализацию агента на моки
       или заглушки без изменения бизнес‑логики.

Интеграция со стеком проекта:
    - Опирается на src.config (Config, LLMSource) для определения активного
      провайдера.
    - Использует src.llm (init_llm, LLMManagerInterface) для создания
      менеджера LLM.
    - Возвращает объект, реализующий интерфейс AgentProtocol, что гарантирует
      единый контракт для всех агентов (health_check, chat).
    - Использует src.logger для логирования выбора и инициализации агента,
      что полезно при деплое и диагностике.

Структура пакета:
    - __init__.py: точка входа, содержит фабрику init_agent и экспортирует
      основные типы и реализации.
    - agent.py: реализация агента (класс Agent), инкапсулирующая логику
      оркестрации RAG‑цикла без привязки к конкретному LLM‑менеджеру.
    - graph.py: граф агента на базе LangGraph для RAG‑цикла.
    - tools.py: фабрика инструментов агента.

Расширяемость:
    Для добавления нового провайдера достаточно:
      1. Создать модуль в пакете src.llm (например, src/llm/mistral_llm.py)
         с реализацией менеджера LLM.
      2. Добавить ветку в init_llm для нового значения LLMSource.
      3. При необходимости добавить новые значения в enum LLMSource.
"""

from src.logger import logger
from src.config import Config, LLMSource
from src.interfaces import (
    AgentProtocol, EmbeddingModelProtocol, VectorDBManagerInterface
)
from src.agent.agent import Agent
from src.agent.tools import make_tools


def init_agent(
    config: Config,
    embedding_model: EmbeddingModelProtocol,
    vector_db: VectorDBManagerInterface,
) -> AgentProtocol:
    """Фабрика для получения реализации главного агента.

    На основе конфигурации выбирает и инициализирует нужный провайдер модели.
    Гарантирует, что будет возвращён объект, реализующий AgentProtocol.

    Args:
        config: Экземпляр конфигурации проекта.
        embedding_model: Модель эмбеддингов (EmbeddingModelProtocol).
        vector_db: Менеджер векторной БД (VectorDBManagerInterface).

    Returns:
        AgentProtocol: Инициализированный агент для работы с выбранной LLM.

    Raises:
        ValueError: Если источник LLM не задан или не поддерживается.
    """
    if config.MAIN_LLM is None:
        raise ValueError(
            f"Не установлен источник LLM: {config.MAIN_LLM_SOURCE}"
        )
    if config.MAIN_LLM_SOURCE == LLMSource.OLLAMA:
        logger.debug("Инициализирован Agent")
        return Agent(config, embedding_model, vector_db)
    raise ValueError(
        f"Неподдерживаемый источник LLM: {config.MAIN_LLM_SOURCE}"
    )


__all__ = ['init_agent', 'make_tools']
