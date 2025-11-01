# Use the official lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency file first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files
COPY . .

# Default environment variables (can be overridden at runtime)
ENV SOURCE_TABLE=practice_records \
    RESULTS_TABLE=dedupe_results \
    LOG_TABLE=dedupe_log \
    THRESHOLD=90 \
    BATCH_SIZE=5000

# Run your main script
CMD ["python", "main.py"]

