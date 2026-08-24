#!/usr/bin/env bash
set -e
cd /app
echo "=== Обновление кода из репозитория ==="
git fetch --depth 1 origin main || true
git reset --hard origin/main || true

echo "=== Проверка зависимостей ==="
pip install --no-cache-dir -r requirements.txt || true

exec python3 /app/start_all.py
