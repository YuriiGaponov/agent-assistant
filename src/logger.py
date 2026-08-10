"""src.logger.py

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

logger.add(
    sink=str(Path(config.LOG_DIR) / 'Agent-Assistant.log'),
    level='DEBUG' if config.DEBUG else config.LOG_LEVEL
)
