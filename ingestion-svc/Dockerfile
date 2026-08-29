FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ingestion-svc has no native binary deps beyond Pillow's wheels
# (which are bundled). No apt packages needed for the service itself.
# Keep the image minimal.

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app/ ./app/
COPY contracts/ ./contracts/
COPY scripts/ ./scripts/
COPY pyproject.toml ./
COPY README.md ./

RUN pip install -e .

EXPOSE 8080

ENV INGESTION_HOST=0.0.0.0 \
    INGESTION_PORT=8080 \
    INGESTION_FAKES3_ROOT=/var/ingestion-fakes3

CMD ["python", "-m", "app.main"]
