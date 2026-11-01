# ── Stage 1: Build dependencies ──────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Production image ────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# fastembed/ONNX Runtime require the OpenMP runtime on slim Debian images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY . .

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create writable runtime directories for the non-root application user.
RUN mkdir -p /app/logs /app/.langgraph_api \
    && adduser --disabled-password --no-create-home --gecos "" appuser \
    && chown -R appuser:appuser /app/logs /app/.langgraph_api

# Don't run as root in production
USER appuser

EXPOSE 8000

# Production server: 4 Uvicorn workers behind Gunicorn process manager
CMD ["uvicorn", "application:application", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--timeout-keep-alive", "30"]
