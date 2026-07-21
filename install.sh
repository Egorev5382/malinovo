#!/bin/bash
set -e

echo "========================================="
echo "  Gate Control — Установка на HA OS"
echo "========================================="

INSTALL_DIR="/opt/gate_system"
cd "$INSTALL_DIR"

echo "[1/5] Виртуальное окружение..."
python3 -m venv venv || python3.11 -m venv venv
source venv/bin/activate

echo "[2/5] Python зависимости..."
pip install --upgrade pip --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
pip install opencv-python-headless SQLAlchemy Flask paho-mqtt PyYAML Pillow numpy --quiet

echo "[3/5] Скачивание моделей..."
mkdir -p models
if [ ! -f models/YOLOS_cars.pt ]; then
    echo "  Клонирование репозитория моделей..."
    git clone --depth 1 https://github.com/smeyanoff/car-number-detection.git /tmp/car-number-detection 2>/dev/null
    cp /tmp/car-number-detection/object_detection/YOLOS_cars.pt models/
    cp /tmp/car-number-detection/lpr_net/model/weights/LPRNet__iteration_2000_28.09.pth models/LPRNet.pth
    rm -rf /tmp/car-number-detection
    echo "  Модели скачаны."
else
    echo "  Модели уже есть."
fi

echo "[4/5] Проверка моделей..."
ls -lh models/

echo "[5/5] Готово!"
echo ""
echo "Настройка: nano $INSTALL_DIR/config.yaml"
echo "Запуск:     cd $INSTALL_DIR && bash run.sh"
echo "Веб:        http://$(hostname -I 2>/dev/null | awk '{print $1}'):8080"
echo "Логин:      admin / gate2024"
