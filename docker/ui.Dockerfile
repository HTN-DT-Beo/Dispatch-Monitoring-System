# ---------- Stage: Streamlit App ----------
FROM python:3.10

# Cài các thư viện hệ thống cần cho xử lý video & GUI
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx libx264-dev \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục làm việc
WORKDIR /app

# Copy requirements.txt vào image
COPY requirements.txt .

# Cài thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn của bạn vào image
COPY Sources/ /app/Sources

# Mở port mặc định của Streamlit
EXPOSE 8501

# Chạy Streamlit app
CMD ["streamlit", "run", "Sources/UI/streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
