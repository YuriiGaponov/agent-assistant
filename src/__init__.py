"""
src.__init__

Корневой пакет проекта Agent‑Assistant.
Предназначен для организации импорта в основной файл main.

Состав:

Модули:
- config: конфигурация проекта
- logger: механизм логирования

Пакеты:
- api: реализация эндпоинтов взаимодействия с пользователем
- RAG: пакет RAG (Retrieval-Augmented Generation)
- vector_db: взаимодействие с векторной БД
"""

from src.api import router
from src.config import get_config
from src.logger import logger
from src.RAG import embedding_model, init_vector_collection
from src.vector_db import get_vectorDB_manager

__all__ = ['get_config', 'logger', 'router', 'embedding_model',
           'init_vector_collection', 'get_vectorDB_manager']
