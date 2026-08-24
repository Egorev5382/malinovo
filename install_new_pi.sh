#!/bin/bash
set -e

INSTALL_DIR="/opt/malinovo"
REPO="https://github.com/Egorev5382/malinovo.git"

echo "=========================================="
echo "  Установка Gate System на новую малинку"
echo "=========================================="

if [ "$EUID" -ne 0 ]; then
    echo "Запусти через sudo: sudo bash $0"
    exit 1
fi

echo "[1/6] Системные пакеты..."
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip git libgl1-mesa-glx libglib2.0-0 > /dev/null 2>&1 || \
apt-get install -y python3-venv python3-pip git libgl1 libglib2.0-0

echo "[2/6] Клонирование репозитория..."
if [ -d "$INSTALL_DIR" ]; then
    cd "$INSTALL_DIR" && git pull || true
else
    git clone "$REPO" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

echo "[3/6] Виртуальное окружение..."
python3 -m venv venv
source venv/bin/activate

echo "[4/6] Python библиотеки (это долго, ~10 мин)..."
pip install --upgrade pip --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
pip install opencv-python-headless SQLAlchemy Flask PyYAML Pillow numpy requests --quiet
pip install easyocr ultralytics --quiet
pip install tqdm --quiet

echo "[5/6] Проверка моделей..."
ls -lh models/YOLOS_cars.pt models/CRNN_int8.pth 2>/dev/null || {
    echo "Моделей нет в git! Скачиваем YOLOS_cars.pt..."
    mkdir -p models
    git clone --depth 1 https://github.com/smeyanoff/car-number-detection.git /tmp/cnd
    cp /tmp/cnd/object_detection/YOLOS_cars.pt models/
    rm -rf /tmp/cnd
}

chown -R $SUDO_USER:$SUDO_USER "$INSTALL_DIR"

echo "[6/6] Готово!"
echo ""
echo "Дальше вручную:"
echo "  1) nano $INSTALL_DIR/config.yaml  — проверь RTSP камеры и токен HA"
echo "  2) sudo bash $(dirname $0)/install_service.sh   — автозапуск 24/7"
echo "  3) Или тест вручную: cd $INSTALL_DIR && source venv/bin/activate && python3 start_all.py"
