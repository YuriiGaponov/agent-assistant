"""src.logger

Модуль настройки логгера на базе Loguru для проекта Agent‑Assistant.
Инициализирует логгер, очищает предыдущую конфигурацию и добавляет
sink для записи логов в файл. Уровень логирования зависит от режима
отладки (DEBUG при DEBUG=True, иначе — LOG_LEVEL из конфига).
"""

from pathlib import Path

from loguru import logger

from src.config import get_config

# Очистка конфигурации логгера
logger.remove()

config = get_config()

# Специализированные типы логов
LOG_TYPES_SPECIAL = ['request']

# Формат записи логов
format = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} "
    "| {name}:{function}:{line} | {message} | {extra}"
)


def user_request_log(record):
    """Фильтр логов обработки запросов пользователей."""
    return record['extra'].get('log_type') == 'request'


def main_log(record):
    """Фильтр логов основного приложения."""
    return record['extra'].get('log_type') not in LOG_TYPES_SPECIAL


# Синк основного лога приложения
logger.add(
    sink=str(Path(config.LOG_DIR) / 'Agent-Assistant.log'),
    level='DEBUG' if config.DEBUG else config.LOG_LEVEL,
    format=format,
    rotation="10 MB",
    retention=7,
    filter=main_log
)

# Синк лога запросов пользователя
logger.add(
    sink=str(Path(config.LOG_DIR) / 'user_requests.log'),
    level=config.LOG_LEVEL,
    format=format,
    rotation="10 MB",
    retention=7,
    filter=user_request_log
)
