#!/bin/bash
set -e
cd "$(dirname "$0")"
source venv/bin/activate

case "$1" in
    camera)  python main.py ;;
    web)     python web_app.py ;;
    all)     python start_all.py ;;
    *)       echo "Использование: $0 {camera|web|all}" ;;
esac
