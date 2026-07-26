FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p reports .cache

ENV APP_ENV=production

# Koyeb (and most container platforms) inject a PORT env var at runtime.
# Using shell form lets us read it; fallback to 8000 for local Docker runs.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
