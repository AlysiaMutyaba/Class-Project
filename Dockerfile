# Use an official PyTorch image with CPU support
FROM pytorch/pytorch:2.2.2-cpu

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy everything else into the container
COPY . .

# Expose Flask port
EXPOSE 5000

# Start the app using gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]