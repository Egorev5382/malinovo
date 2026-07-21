#!/bin/bash
set -e

echo "========================================="
echo "  Gate Control — Установка на RPi 5"
echo "========================================="

INSTALL_DIR="/opt/gate_system"

echo "[1/7] Системные зависимости..."
sudo apt update && sudo apt install -y python3 python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 git

echo "[2/7] Копирование проекта..."
sudo mkdir -p $INSTALL_DIR
sudo cp -r . $INSTALL_DIR/
cd $INSTALL_DIR

echo "[3/7] Виртуальное окружение..."
python3 -m venv venv
source venv/bin/activate

echo "[4/7] Python зависимости..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python-headless SQLAlchemy Flask paho-mqtt PyYAML Pillow numpy

echo "[5/7] Скачивание моделей..."
mkdir -p models
if [ ! -f models/YOLOS_cars.pt ]; then
    git clone --depth 1 https://github.com/smeyanoff/car-number-detection.git /tmp/car-number-detection
    cp /tmp/car-number-detection/object_detection/YOLOS_cars.pt models/
    cp /tmp/car-number-detection/lpr_net/model/weights/LPRNet__iteration_2000_28.09.pth models/LPRNet.pth
    rm -rf /tmp/car-number-detection
fi

echo "[6/7] Systemd сервисы..."
sudo tee /etc/systemd/system/gate-camera.service > /dev/null << EOF
[Unit]
Description=Gate Control - Camera
After=network.target
[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python main.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/gate-web.service > /dev/null << EOF
[Unit]
Description=Gate Control - Web
After=network.target
[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python web_app.py
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1
[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable gate-camera gate-web

echo "[7/7] Готово!"
echo ""
echo "Настройка: nano $INSTALL_DIR/config.yaml"
echo "Запуск:     sudo systemctl start gate-camera gate-web"
echo "Веб:        http://$(hostname -I | awk '{print $1}'):8080"
echo "Логин:      admin / gate2024"
