# Use Python 3.10 slim as a base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY main.py ./
COPY sheets_sync/ ./sheets_sync/

# Set default environment variables
ENV SOURCE_TABLE=practice_records \
    RESULTS_TABLE=dedupe_results \
    LOG_TABLE=dedupe_log \
    THRESHOLD=90 \
    BATCH_SIZE=5000

# Run the application
CMD ["python", "main.py"]
