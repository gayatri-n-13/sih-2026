# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

# GDAL/rasterio needs libgdal + its headers at runtime.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgdal32 libgeos-dev libproj-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/ingestion

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY pyproject.toml ./

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
