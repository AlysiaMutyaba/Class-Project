# Use official Python 3.12 slim image (matches local training environment)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make entrypoint script executable (created below)
# RUN chmod +x ./wait-for-db.sh || true

# Expose port
EXPOSE 5000

# Entrypoint will wait for DB then exec the CMD
# ENTRYPOINT ["./wait-for-db.sh"]

# Default command to run the app via gunicorn with memory optimizations
# Use 1 worker to reduce memory footprint (512MB free tier)
# --timeout 300 for slow model loading on first request
# --max-requests 100 to recycle worker and prevent memory leaks
# --preload for loading model once before forking
CMD ["gunicorn", "-w", "1", "--threads", "2", "-b", "0.0.0.0:5000", "--timeout", "300", "--max-requests", "100", "--max-requests-jitter", "20", "--preload", "app:app"]