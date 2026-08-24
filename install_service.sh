#!/bin/bash
set -e

INSTALL_DIR="/opt/malinovo"
USER_NAME="${SUDO_USER:-pi}"

if [ "$EUID" -ne 0 ]; then
    echo "Запусти через sudo: sudo bash $0"
    exit 1
fi

cat > /etc/systemd/system/gate-system.service << EOF
[Unit]
Description=Gate Control System (распознавание номеров)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python3 $INSTALL_DIR/start_all.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/gate-system.log
StandardError=append:/var/log/gate-system.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable gate-system
systemctl restart gate-system

echo "=== Сервис установлен и запущен ==="
echo ""
echo "Команды:"
echo "  systemctl status gate-system    — статус"
echo "  systemctl restart gate-system   — перезапуск"
echo "  systemctl stop gate-system      — остановка"
echo "  journalctl -u gate-system -f    — логи в реальном времени"
echo "  tail -f /var/log/gate-system.log — логи из файла"
