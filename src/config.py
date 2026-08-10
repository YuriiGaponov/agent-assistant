"""src.config.py

Модуль конфигурации проекта Agent‑Assistant.
Предоставляет класс Config на базе Pydantic Settings
и функцию get_config() для получения единого экземпляра
конфигурации с кэшированием. Загружает параметры
из .env-файла, соответствующего ENVIRONMENT.
"""


import os
from enum import Enum
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
"""Корневой путь проекта — директория, расположенная на уровень выше src."""

ENV_MODE = os.getenv('ENVIRONMENT', 'development')
"""Режим окружения: из ENVIRONMENT или 'development' по умолчанию."""


class LogLevel(str, Enum):
    """Допустимые уровни логирования."""

    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'


class VectorDB(str, Enum):
    """Поддерживаемые векторные БД."""
    CHROMA = 'chroma'


class LLMSource(str, Enum):
    """Поддерживаемые источники LLM."""
    OLLAMA = 'ollama'


class LLM(str, Enum):
    """Поддерживаемые LLM."""
    mistral = 'mistral'
    qwen = 'qwen2.5'


class Config(BaseSettings):
    """
    Класс для централизованного управления конфигурацией проекта.
    Считывает параметры из файла .env и переменных окружения.
    """

    # === Описание ===
    TITLE: str = 'Agent-Assistant'
    """str: Название приложения."""

    DESCRIPTION: str = (
        'Проект AI‑агента, для работы с внутренней документацией.'
    )
    """str: Краткое описание приложения."""

    VERSION: str = '0.1_MVP'
    """str: Версия приложения."""

    # === Режим отладки ===
    DEBUG: bool = False
    """bool: Флаг режима отладки приложения. По умолчанию — False."""

    # === Настройки логирования ===
    LOG_DIR: Path = BASE_DIR / 'logs'
    """Path: Директория для логов приложения. По умолчанию — 'logs'."""

    LOG_LEVEL: LogLevel = LogLevel.INFO
    """LogLevel: Уровень логирования. По умолчанию — INFO."""

    # === Настройки агентов ===
    MAIN_LLM_SOURCE: LLMSource | None = None
    """Источник главной LLM, ОБЯЗАТЕЛЬНО должен быть определен в .env,
    значение по умолчанию None прерывает запуск всего приложения.
    """

    MAIN_LLM: LLM | None = None
    """Главная LLM, ОБЯЗАТЕЛЬНО должен быть определен в .env,
    значение по умолчанию None прерывает запуск всего приложения.
    """

    # === Настройки RAG ===
    DOC_DIR: Path = BASE_DIR / 'docs'
    """Path: Директория для внутренней документации. По умолчанию — 'docs'."""

    EMBEDDING_MODEL: str = 'all-MiniLM-L6-v2'
    """str: Модель для генерации эмбеддингов."""

    # === Настройки векторной БД ===
    VECTOR_DB: VectorDB = VectorDB.CHROMA
    """VectorDB: Векторная БД."""

    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / f".env.{ENV_MODE}"),
        env_file_encoding='utf-8',
        extra='ignore',
        populate_by_name=True,
    )


@lru_cache
def get_config() -> Config:
    """
    Возвращает глобальный экземпляр конфигурации приложения.

    Результат кэшируется с помощью lru_cache, поэтому Config()
    создаётся только при первом вызове. Последующие вызовы
    возвращают закэшированный экземпляр.

    При тестировании легко заменяется на мок.

    Returns:
        Config: Настроенный экземпляр класса конфигурациии приложения.
    """
    return Config()
