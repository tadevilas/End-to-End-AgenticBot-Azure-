# syntax=docker/dockerfile:1.4
# ==============================================================================
# Multi-Stage Production Dockerfile for Enterprise Agentic RAG Platform
# Supports targets: 'backend' (FastAPI) and 'frontend' (Streamlit)
# ==============================================================================

# --- Stage 1: Base Image ---
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/cache/huggingface \
    FASTEMBED_CACHE_PATH=/tmp/cache/fastembed

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Stage 2: Builder (Compile dependencies) ---
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# --- Stage 3: Backend Runtime (FastAPI) ---
FROM base AS backend

# Create non-root system user and directories
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser \
    && mkdir -p /tmp/cache/huggingface /tmp/cache/fastembed /app \
    && chown -R appuser:appgroup /tmp/cache /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source code
COPY --chown=appuser:appgroup src/ /app/src/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

# --- Stage 4: Frontend Runtime (Streamlit) ---
FROM base AS frontend

# Create non-root system user and directories
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /sbin/nologin appuser \
    && mkdir -p /app \
    && chown -R appuser:appgroup /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy frontend source code
COPY --chown=appuser:appgroup frontend/ /app/frontend/

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
