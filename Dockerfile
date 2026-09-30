# ============================================================
# Dockerfile для Agent-Assistant
# FastAPI-приложение с RAG-агентом
# ============================================================

# ---- Базовый образ ----
FROM python:3.13.15-slim

# ---- Рабочая директория ----
WORKDIR /app

# ---- Системные зависимости ----
RUN apt-get update && apt-get install -y libmagic1 && rm -rf /var/lib/apt/lists/*

# ---- Установка зависимостей Python ----
# Слой кэшируется: при изменении только исходного кода
# зависимости переустанавливаться не будут.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Предзагрузка модели эмбеддингов в кэш образа
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ---- Копирование исходного кода и конфигурации ----
COPY . .

# Делаем entrypoint.sh исполняемым
RUN chmod +x entrypoint.sh

# ---- Создание директорий под данные ----
# Содержимое будет монтироваться снаружи через volumes.
RUN mkdir -p /app/database /app/docs /app/logs

# ---- Порт HTTP-приложения ----
EXPOSE 8000

# ---- Запуск: миграции БД + ASGI-сервер ----
ENTRYPOINT ["./entrypoint.sh"]