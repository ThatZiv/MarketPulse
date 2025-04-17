FROM python:3.12-slim

WORKDIR /app

RUN mkdir -p /app/llm

RUN cd /app && apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    postgresql-client \
    libffi-dev \
    libssl-dev \
    build-essential \
    curl \
    git

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade pip

RUN python -m venv ./.venv && . ./.venv/bin/activate
RUN pip install -r requirements.txt

EXPOSE 5000



