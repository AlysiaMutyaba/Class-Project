# Use official Python 3.10 slim image
FROM python:3.10-slim

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

# Default command to run the app via gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]