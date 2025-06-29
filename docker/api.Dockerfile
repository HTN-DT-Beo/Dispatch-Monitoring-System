FROM python:3.11

# Cài ffmpeg + unzip + gdown
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libavcodec-extra \
    unzip \
    && apt-get clean

# Cài gdown (dùng để tải từ Google Drive)
RUN pip install --no-cache-dir gdown

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copy requirements và cài
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY Sources/ /app/Sources

# Copy script tải model và thực thi
COPY download_models.sh /app/download_models.sh
RUN chmod +x /app/download_models.sh && ./download_models.sh

# Cấu hình
ENV PYTHONPATH=/app
EXPOSE 8000
RUN mkdir /app/videos

# Chạy app
CMD ["uvicorn", "Sources.API.main:app", "--host", "0.0.0.0", "--port", "8000"]