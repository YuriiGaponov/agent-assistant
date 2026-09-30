#!/bin/sh
set -e

# Запускаем сервер в фоне
ollama serve &
OLLAMA_PID=$!

# Ждём, пока сервер поднимется
echo "Ожидание запуска Ollama..."
until ollama list >/dev/null 2>&1; do
  sleep 2
done
echo "Ollama готова."

# Тянем модель (если уже скачана — просто сверит хэш)
echo "Загрузка модели qwen2.5..."
ollama pull qwen2.5
echo "Модель готова."

# Передаём управление на сервер
wait $OLLAMA_PID
