FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p models photos \
    && git clone --depth 1 https://github.com/smeyanoff/car-number-detection.git /tmp/car-number-detection \
    && cp /tmp/car-number-detection/object_detection/YOLOS_cars.pt models/ \
    && cp /tmp/car-number-detection/lpr_net/model/weights/LPRNet__iteration_2000_28.09.pth models/LPRNet.pth \
    && rm -rf /tmp/car-number-detection

EXPOSE 8080

CMD ["python", "start_all.py"]
