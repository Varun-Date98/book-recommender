FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REDIS_URL=redis://127.0.0.1:6379/0

# Install redis-server, redis-cli, tini (for proper signal handling) and curl (for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
      redis-server redis-tools tini curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Backend-only deps (no Streamlit here)
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Your backend code
COPY . /app/src/backend

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD curl -fsS http://localhost:8000/heart_beat || exit 1

CMD ["python", "-m", "uvicorn", "src.backend.recommender_api:app", "--host", "0.0.0.0", "--port", "8000"]
