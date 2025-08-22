# Dockerfile
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# OS deps (gcc pour llama-cpp-python wheel si besoin; libgomp pour BLAS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install -e .[dev]

COPY . .
# Crée les dossiers pour monter les volumes
RUN mkdir -p /models /hf_cache

EXPOSE 8000
ENV MODELS_DIR=/models HF_HOME=/hf_cache

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "slm_api.app:app"]