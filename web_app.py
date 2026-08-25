import os
import glob
import yaml
import secrets
import functools
import logging
from flask import (Flask, render_template, request, redirect, url_for,
                   flash, send_from_directory, jsonify, session)
from database import Database
from data_dir import get_data_dir, resolve_db_path, migrate_old_data

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))
app.secret_key = secrets.token_hex(32)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

ADMIN_USER = "admin"
ADMIN_PASS = "gate2024"


def load_config(path=None):
    if path is None:
        path = os.path.join(BASE_DIR, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = load_config()
DATA_DIR = get_data_dir()
migrate_old_data(DATA_DIR)
db = Database(db_path=resolve_db_path(_config["database"]["path"], DATA_DIR),
              photos_dir=os.path.join(DATA_DIR, "photos"))


def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == ADMIN_USER and password == ADMIN_PASS:
            session["logged_in"] = True
            session["username"] = username
            flash("Вход выполнен", "success")
            return redirect(url_for("logs"))
        else:
            flash("Неверный логин или пароль", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Вы вышли из системы", "info")
    return redirect(url_for("login"))


@app.route("/")
@login_required
def logs():
    page = request.args.get("page", 1, type=int)
    per_page = 20
    offset = (page - 1) * per_page
    logs_data = db.get_logs(limit=per_page, offset=offset)
    total = db.get_log_count()
    total_pages = (total + per_page - 1) // per_page
    return render_template("index.html", logs=logs_data,
                           page=page, total_pages=total_pages, total=total)


@app.route("/plates")
@login_required
def plates():
    all_plates = db.get_all_plates()
    return render_template("plates.html", plates=all_plates)


@app.route("/plates/add", methods=["POST"])
@login_required
def add_plate():
    plate = request.form.get("plate", "").upper().strip()
    owner = request.form.get("owner", "").strip()
    if not plate:
        flash("Введите номерной знак", "warning")
        return redirect(url_for("plates"))
    if db.add_plate(plate, owner):
        flash(f"Номер {plate} добавлен", "success")
    else:
        flash(f"Номер {plate} уже существует", "warning")
    return redirect(url_for("plates"))


@app.route("/plates/remove", methods=["POST"])
@login_required
def remove_plate():
    plate = request.form.get("plate", "").strip()
    if db.remove_plate(plate):
        flash(f"Номер {plate} удалён", "success")
    else:
        flash(f"Номер {plate} не найден", "warning")
    return redirect(url_for("plates"))


@app.route("/view/<int:log_id>")
@login_required
def view_entry(log_id):
    session_db = db.Session()
    try:
        from database import EntryLog
        log_entry = session_db.query(EntryLog).filter_by(id=log_id).first()
        if log_entry:
            return render_template("view.html", log={
                "id": log_entry.id,
                "plate": log_entry.plate,
                "detected_at": log_entry.detected_at.isoformat(),
                "photo_path": log_entry.photo_path,
                "allowed": log_entry.allowed,
                "gate_opened": log_entry.gate_opened,
                "confidence": log_entry.confidence
            })
    finally:
        session_db.close()
    flash("Запись не найдена", "warning")
    return redirect(url_for("logs"))


@app.route("/photos/<path:filename>")
@login_required
def photos(filename):
    return send_from_directory(os.path.join(DATA_DIR, "photos"), filename)


@app.route("/api/logs")
@login_required
def api_logs():
    limit = request.args.get("limit", 50, type=int)
    logs_data = db.get_logs(limit=limit)
    return jsonify(logs_data)


@app.route("/api/plates")
@login_required
def api_plates():
    return jsonify(db.get_all_plates())


@app.route("/api/plates/add", methods=["POST"])
@login_required
def api_add_plate():
    data = request.get_json()
    plate = data.get("plate", "").upper().strip()
    owner = data.get("owner", "")
    if not plate:
        return jsonify({"error": "Номер обязателен"}), 400
    success = db.add_plate(plate, owner)
    return jsonify({"success": success, "plate": plate})


@app.route("/api/plates/remove", methods=["POST"])
@login_required
def api_remove_plate():
    data = request.get_json()
    plate = data.get("plate", "")
    success = db.remove_plate(plate)
    return jsonify({"success": success})


@app.route("/api/gate/open", methods=["POST"])
@login_required
def api_open_gate():
    config = load_config()
    use_ha = config.get("gate", {}).get("use_ha", False)
    if use_ha:
        from ha_gate import HAGate
        ha_cfg = config.get("homeassistant", {})
        gate = HAGate(
            entity_id=ha_cfg.get("entity_id", "switch.vorota"),
            ha_url=ha_cfg.get("ha_url") or None,
            ha_token=ha_cfg.get("ha_token") or None
        )
        gate.connect()
        result = gate.open_gate()
        gate.disconnect()
    else:
        from mqtt_gate import MQTTGate
        gate = MQTTGate(
            broker=config["mqtt"]["broker"],
            port=config["mqtt"]["port"],
            topic=config["mqtt"]["topic"],
            username=config["mqtt"].get("username", ""),
            password=config["mqtt"].get("password", "")
        )
        gate.connect()
        import time
        time.sleep(0.5)
        result = gate.open_gate()
        gate.disconnect()
    return jsonify({"success": result})


@app.route("/startup")
def startup():
    startup_photo = os.path.join(BASE_DIR, "startup_photo.jpg")
    has_photo = os.path.exists(startup_photo)
    return render_template("startup.html", has_photo=has_photo)


@app.route("/startup/confirm", methods=["POST"])
def startup_confirm():
    flag_file = os.path.join(BASE_DIR, "system_started")
    with open(flag_file, "w") as f:
        f.write("ok")
    flash("Система запущена!", "success")
    return redirect(url_for("logs"))


@app.route("/startup/photo")
def startup_photo():
    return send_from_directory(BASE_DIR, "startup_photo.jpg")


@app.route("/snapshots")
@login_required
def snapshots():
    snaps = sorted(glob.glob(os.path.join(BASE_DIR, "snap_*.jpg")), reverse=True)
    snap_files = [os.path.basename(s) for s in snaps]
    return render_template("snapshots.html", snapshots=snap_files)


@app.route("/snapshots/<path:filename>")
@login_required
def serve_snapshot(filename):
    return send_from_directory(BASE_DIR, filename)


if __name__ == "__main__":
    db = Database(
        db_path=resolve_db_path(load_config()["database"]["path"], DATA_DIR),
        photos_dir=os.path.join(DATA_DIR, "photos")
    )
    app.run(
        host=load_config()["web"]["host"],
        port=load_config()["web"]["port"],
        debug=False
    )
