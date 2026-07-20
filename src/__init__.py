"""
src.__init__

Корневой пакет проекта Agent‑Assistant.
Предназначен для организации импорта в основной файл main.
"""

from src.config import get_config
from src.logger import logger

__all__ = ['get_config', 'logger']
