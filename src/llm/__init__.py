"""src.llm.__init__

Пакет реализации менеджеров LLM в проекте Agent‑Assistant.

Предоставляет:
    - Фабрику init_llm для создания менеджера LLM в зависимости от
      выбранного источника LLM (Ollama, Mistral, OpenRouter и т.д.).
    - Изоляцию низкоуровневой логики взаимодействия с моделью от пакета
      агента: пакет src.agent отвечает за оркестрацию (граф), а здесь
      сосредоточено всё, что касается непосредственной работы с моделью.
    - Единый контракт через интерфейс LLMManagerInterface, гарантирующий
      наличие обязательных методов (ainvoke, bind_tools, health_check)
      у любого менеджера.

Структура пакета:
    - __init__.py: точка входа, содержит фабрику init_llm и экспортирует
      основные типы и реализации.
    - base.py: базовый класс менеджера LLM (BaseLLMManager).
    - ollama_llm.py: реализация менеджера на базе Ollama (OllamaLLMManager).

Расширяемость:
    Для добавления нового провайдера достаточно:
      1. Создать модуль (например, src/llm/mistral_llm.py) с реализацией
         менеджера, который наследует BaseLLMManager.
      2. Добавить ветку в init_llm для нового значения LLMSource.
      3. При необходимости добавить новые значения в enum LLMSource.
"""

from src.config import Config, LLMSource
from src.logger import logger
from src.interfaces import LLMManagerInterface
from src.llm.ollama_llm import OllamaLLMManager


def init_llm(config: Config) -> LLMManagerInterface:
    """Фабрика для получения реализации менеджера LLM.

    На основе конфигурации выбирает и инициализирует нужного провайдера.
    Гарантирует, что будет возвращён объект, реализующий LLMManagerInterface.

    Args:
        config: Экземпляр конфигурации проекта.

    Returns:
        LLMManagerInterface: Инициализированный менеджер LLM.

    Raises:
        ValueError: Если источник LLM не задан или не поддерживается.
    """
    if config.MAIN_LLM is None:
        raise ValueError(
            f"Не установлен источник LLM: {config.MAIN_LLM_SOURCE}"
        )
    if config.MAIN_LLM_SOURCE == LLMSource.OLLAMA:
        logger.debug("Инициализирован OllamaLLMManager")
        return OllamaLLMManager(config)
    raise ValueError(
        f"Неподдерживаемый источник LLM: {config.MAIN_LLM_SOURCE}"
    )


__all__ = ['init_llm', 'OllamaLLMManager']
