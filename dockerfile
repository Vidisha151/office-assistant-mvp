FROM python:3.10-slim

WORKDIR /app

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libomp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-dateutil pydantic joblib python-multipart dateparser pandas scikit-learn
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir transformers sentence-transformers faiss-cpu tqdm

# Project files
COPY backend/ /app/backend/
COPY intents.csv /app/intents.csv
COPY frontend/ /app/frontend_build/

WORKDIR /app/backend

ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
