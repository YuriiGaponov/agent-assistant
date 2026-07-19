"""main.py

Точка входа FastAPI‑приложения.
Инициализирует экземпляр приложения app и служит основой для регистрации
роутеров и настройки зависимостей.
"""

from fastapi import FastAPI

from src import get_config

config = get_config()

app = FastAPI(
    debug=config.DEBUG,
    title=config.TITLE,
    description=config.DESCRIPTION,
    version=config.VERSION
)
