# Dockerfile for AI-Enhanced Deduplication Pipeline
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY sheets_sync/ ./sheets_sync/
COPY dedupe_logic/ ./dedupe_logic/
COPY database/ ./database/

# Set Python to run in unbuffered mode for better logging
ENV PYTHONUNBUFFERED=1

# Optimize torch for CPU
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

# Run the application
CMD ["python", "main.py"]
