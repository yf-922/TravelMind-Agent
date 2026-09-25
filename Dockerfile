FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .

RUN useradd --create-home --uid 10001 appuser && \
    mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=3)"

CMD ["python", "run.py"]
