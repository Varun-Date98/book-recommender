FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install frontend deps
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the frontend code
COPY  . /app/src/frontend

# API Base URL
ENV BASE_URL=http://api:8000

EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
CMD ["python","-m","streamlit","run","src/frontend/app.py","--server.address=0.0.0.0","--server.port","8501"]