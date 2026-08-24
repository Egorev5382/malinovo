#!/usr/bin/env bash
set -e
cd /app
echo "=== Обновление кода из репозитория ==="
git fetch --depth 1 origin main || true
git reset --hard origin/main || true
exec python3 /app/start_all.py
