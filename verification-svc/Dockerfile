FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the src directory is in PYTHONPATH
ENV PYTHONPATH="/app/src"

EXPOSE 50051

CMD ["python", "src/verification_svc/server.py"]
