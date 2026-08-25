FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth 1 https://github.com/Egorev5382/malinovo.git /app

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    opencv-python-headless SQLAlchemy Flask PyYAML Pillow numpy requests \
    pandas matplotlib seaborn tqdm easyocr ultralytics paho-mqtt

ENV EASYOCR_MODULE_PATH=/opt/easyocr_models
RUN mkdir -p /opt/easyocr_models \
    && python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
    && mv yolov8n.pt /opt/yolov8n.pt \
    && python -c "import easyocr; easyocr.Reader(['ru','en'], gpu=False, verbose=False)"

COPY run.sh /run.sh
RUN chmod +x /run.sh

CMD ["/run.sh"]
