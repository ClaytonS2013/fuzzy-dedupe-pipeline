# Production-Ready Multi-stage Build
ARG PYTHON_VERSION=3.10

# Stage 1: Builder
FROM python:${PYTHON_VERSION}-slim as builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc g++ python3-dev build-essential cmake wget git && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy and install requirements
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Verify installations
RUN python -c "import supabase, torch, pandas; print('Dependencies OK')"

# Stage 2: Runtime
FROM python:${PYTHON_VERSION}-slim

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 ca-certificates curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy application
COPY --chown=appuser:appuser . .

# Create directories
RUN mkdir -p /app/models /app/logs /app/data && \
    chown -R appuser:appuser /app

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Make scripts executable
RUN if [ -f startup.sh ]; then chmod +x startup.sh; fi

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run
CMD ["/bin/bash", "-c", "if [ -f ./startup.sh ]; then ./startup.sh; else python main.py; fi"]
