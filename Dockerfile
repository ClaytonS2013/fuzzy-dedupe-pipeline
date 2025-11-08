ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim as builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc g++ python3-dev build-essential cmake wget git && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir --upgrade sentence-transformers==2.2.2 faiss-cpu==1.7.4

# Verify AI packages
RUN python -c "import sentence_transformers; import faiss; print('✅ AI packages installed')"

FROM python:${PYTHON_VERSION}-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 ca-certificates curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY --chown=appuser:appuser . .

RUN mkdir -p /app/models /app/logs /app/data /app/config /app/database && \
    chown -R appuser:appuser /app && \
    if [ -f startup.sh ]; then chmod +x startup.sh; fi && \
    if [ -f fix_ai.sh ]; then chmod +x fix_ai.sh; fi

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TRANSFORMERS_CACHE=/app/models

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import sentence_transformers, faiss; print('AI OK')"

CMD ["/bin/bash", "-c", "if [ -f ./startup.sh ]; then ./startup.sh; else python main.py; fi"]
