FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for rasterio/GDAL wheels (kept minimal for headless).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgeos-dev \
        libproj-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY preprocessing_svc/ ./preprocessing_svc/
COPY mock_ingestion/ ./mock_ingestion/
COPY pyproject.toml ./
COPY README.md ./

RUN pip install -e .

EXPOSE 8080

ENV PREPROC_HOST=0.0.0.0 \
    PREPROC_PORT=8080 \
    PREPROC_OUTPUT_DIR=/var/outputs

CMD ["python", "-m", "preprocessing_svc.api"]
