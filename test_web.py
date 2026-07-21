import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from web_app import app, db

print("=" * 50)
print("  Gate Control - Тестовый запуск")
print("=" * 50)

test_plates = [
    ("A123BC77", "Иванов И.И."),
    ("B456KK99", "Петров П.П."),
    ("C789MM52", "Сидоров С.С."),
]
for plate, owner in test_plates:
    db.add_plate(plate, owner)

test_logs = [
    ("A123BC77", True, True, 0.95),
    ("X999YY77", False, False, 0.88),
    ("B456KK99", True, True, 0.92),
    ("K111AB50", False, False, 0.76),
    ("C789MM52", True, True, 0.91),
]
for plate, allowed, gate_opened, conf in test_logs:
    db.add_log(plate=plate, photo_path="", allowed=allowed, gate_opened=gate_opened, confidence=conf)

print()
print("  Тестовые данные добавлены")
print()
print("=" * 50)
print("  Открой в браузере: http://localhost:8080")
print("  Логин: admin")
print("  Пароль: gate2024")
print("=" * 50)

app.run(host="0.0.0.0", port=8080, debug=False)
