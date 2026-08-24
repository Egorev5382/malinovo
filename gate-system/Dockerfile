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

COPY run.sh /run.sh
RUN chmod +x /run.sh

CMD ["/run.sh"]
