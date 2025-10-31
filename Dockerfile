FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency definitions and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY dedupe_pipeline.py ./
COPY main.py ./

# Default environment variables (can be overridden at runtime)
ENV SOURCE_TABLE=practice_records \
    RESULTS_TABLE=dedupe_results \
    LOG_TABLE=dedupe_log \
    THRESHOLD=90 \
    BATCH_SIZE=5000

# Define the entrypoint for Railway scheduled runs
CMD ["python", "main.py"]
