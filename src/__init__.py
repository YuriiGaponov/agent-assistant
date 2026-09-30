"""src.__init__

Корневой пакет проекта Agent‑Assistant.
Предназначен для организации импорта в основной файл main.

Состав:

Модули:
- config: конфигурация проекта
- logger: механизм логирования
- interfaces: контракты модулей, пакетов, внешних систем
- health_check_service: проверка работоспособности

Пакеты:
- api: реализация эндпоинтов взаимодействия с пользователем
- agent: взаимодействие с ИИ‑агентом (граф, оркестрация)
- db: взаимодействие с реляционной БД
- llm: менеджеры LLM (низкоуровневая работа с моделью)
- middleware: middleware ПО
- RAG: пакет RAG (Retrieval-Augmented Generation)
- vector_db: взаимодействие с векторной БД
"""

from src.api import router
from src.config import get_config
from src.logger import logger
from src.RAG import embedding_model, init_vector_collection
from src.db import get_async_engine, create_session_maker
from src.vector_db import get_vectorDB_manager
from src.agent import init_agent
from src.middleware import RequestLoggingMiddleware

__all__ = ['get_config', 'logger', 'router', 'embedding_model',
           'init_vector_collection', 'get_vectorDB_manager',
           'init_agent', 'RequestLoggingMiddleware', 'get_async_engine',
           'create_session_maker']
