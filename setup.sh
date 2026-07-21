#!/bin/bash
set -e
echo "=== Gate Control — Установка ==="

if ! command -v python3 &>/dev/null; then
    echo "[1] Скачивание Python 3.11..."
    cd /tmp
    wget -q https://github.com/indygreg/python-build-standalone/releases/download/20240415/cpython-3.11.9+20240415-aarch64-unknown-linux-gnu-install_only.tar.gz -O py.tar.gz
    mkdir -p /opt/python
    tar xzf /tmp/py.tar.gz -C /opt/python
    rm /tmp/py.tar.gz
    export PATH="/opt/python/bin:$PATH"
    echo 'export PATH="/opt/python/bin:$PATH"' >> ~/.bashrc
    echo "Python установлен: $(/opt/python/bin/python3 --version)"
else
    echo "[1] Python уже есть: $(python3 --version)"
fi

export PATH="/opt/python/bin:$PATH"
PYTHON=$(command -v python3 || echo /opt/python/bin/python3)

echo "[2] Клонирование проекта..."
cd /opt
rm -rf gate_system
git clone --depth 1 https://github.com/Egorev5382/malinovo.git gate_system
cd gate_system

echo "[3] Виртуальное окружение..."
$PYTHON -m venv venv
source venv/bin/activate

echo "[4] Установка зависимостей..."
pip install --upgrade pip --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
pip install opencv-python-headless SQLAlchemy Flask paho-mqtt PyYAML Pillow numpy --quiet

echo "[5] Скачивание моделей..."
mkdir -p models
git clone --depth 1 https://github.com/smeyanoff/car-number-detection.git /tmp/cnd
cp /tmp/cnd/object_detection/YOLOS_cars.pt models/
cp /tmp/cnd/lpr_net/model/weights/LPRNet__iteration_2000_28.09.pth models/LPRNet.pth
rm -rf /tmp/cnd

echo "=== Готово! ==="
echo "Настройка: nano /opt/gate_system/config.yaml"
echo "Запуск:     cd /opt/gate_system && bash run.sh all"
