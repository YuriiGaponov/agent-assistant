"""src.config.py

Модуль конфигурации проекта Agent‑Assistant.
Предоставляет класс Config на базе Pydantic Settings
и функцию get_config() для получения единого экземпляра
конфигурации с кэшированием. Загружает параметры
из .env-файла, соответствующего ENVIRONMENT.
"""


import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
"""Корневой путь проекта — директория, расположенная на уровень выше src."""

ENV_MODE = os.getenv('ENVIRONMENT', 'development')
"""Режим окружения: из ENVIRONMENT или 'development' по умолчанию."""


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

    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / f".env.{ENV_MODE}"),
        env_file_encoding='utf-8',
        extra='ignore'
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
