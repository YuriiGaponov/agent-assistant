"""
src.__init__

Корневой пакет проекта Agent‑Assistant.
Предназначен для организации импорта в основной файл main.
"""

from src.config import get_config

__all__ = ['get_config']
