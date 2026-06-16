# ===========================================================================
# SubAI Backend — Dockerfile
# ===========================================================================
# Python 3.11 slim tabanlı, ffmpeg ve Türkçe karakter desteği dahil.
# Render.com'da deploy için hazır (port 10000).
# ===========================================================================

FROM python:3.11-slim

# Sistem bağımlılıkları — ffmpeg ve font desteği
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-noto \
        fonts-noto-core \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Font cache güncelle (ASS altyazıları için gerekli)
RUN fc-cache -fv

# Çalışma dizini
WORKDIR /app

# Önce bağımlılıkları kur (Docker cache optimizasyonu)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

# Geçici dosyalar için dizin oluştur
RUN mkdir -p /tmp/subai

# Render.com varsayılan portu
EXPOSE 10000

# Sağlık kontrolü
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10000/api/health')" || exit 1

# Uygulamayı başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
