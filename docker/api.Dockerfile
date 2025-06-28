FROM python:3.11

# 0. Cài ffmpeg (đọc/ghi mp4) + tini (quản lý signal gọn)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libavcodec-extra \
    && apt-get clean

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 1. Copy & cài requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy code + models + botsort yaml
COPY Sources/  /app/Sources
COPY Models/  /app/Models


# 3. Khai báo port & PYTHONPATH
ENV PYTHONPATH=/app
EXPOSE 8000

# 4. Tạo thư mục videos để lưu kết quả
RUN mkdir /app/videos

# 5. Run Uvicorn
CMD ["uvicorn", "Sources.API.main:app", "--host", "0.0.0.0", "--port", "8000"]


