# Use a stable Python base
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install system dependencies needed for FastAI, PyTorch, SciPy, Matplotlib, image libraries, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Expose the port your Flask app uses
EXPOSE 5000

# Command to run the Flask app with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]