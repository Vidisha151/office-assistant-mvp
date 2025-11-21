# Python base
FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libomp-dev \
    pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Install Python packages
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-dateutil pydantic joblib python-multipart dateparser pandas scikit-learn openai
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir transformers sentence-transformers faiss-cpu tqdm

# Copy backend
COPY backend/ /app/backend/
COPY frontend/ /app/frontend_build/

WORKDIR /app/backend
ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
