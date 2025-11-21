# Dockerfile — Python 3.10, CPU-only ML stack (torch, transformers, sentence-transformers, faiss-cpu)
# Build a large image suitable for running sentence-transformers + faiss in CPU mode.

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# -----------------------
# 1) System packages
# -----------------------
# install build tools and libs needed by numpy/torch/faiss
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    wget \
    git \
    curl \
    unzip \
    libsndfile1 \
    libopenblas-dev \
    liblapack-dev \
    libomp-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# 2) Python tooling
# -----------------------
RUN python -m pip install --upgrade pip setuptools wheel

# -----------------------
# 3) Install small, fast packages first (cacheable)
# -----------------------
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-dateutil \
    pydantic \
    joblib \
    python-multipart \
    dateparser \
    pandas \
    scikit-learn

# -----------------------
# 4) Install PyTorch (CPU-only) — use official PyTorch CPU index
# -----------------------
# This will pull a CPU wheel compatible with python3.10 on Linux x86_64.
# If you want a specific torch version, replace "torch" with "torch==2.2.0+cpu" and keep index-url.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# -----------------------
# 5) Install transformers & sentence-transformers
# -----------------------
RUN pip install --no-cache-dir transformers sentence-transformers

# -----------------------
# 6) Install faiss-cpu (may be large)
# -----------------------
RUN pip install --no-cache-dir faiss-cpu

# -----------------------
# 7) Any other python libs you want
# -----------------------
RUN pip install --no-cache-dir tqdm

# -----------------------
# 8) Copy project files
# -----------------------
# Copy backend code first (so pip layers can cache separately in future builds)
COPY backend/ ./backend/
COPY backend/requirements.txt ./backend/requirements.txt
# If you have a pretrained intent_model.pkl, copy it:

COPY intents.csv ./intents.csv

# Copy frontend static files into place expected by backend
COPY frontend/ ./frontend_build/

# Move into backend when running; optionally adjust if your app module is backend/app.py
WORKDIR /app/backend

# Expose port and run via python -m uvicorn to avoid PATH issues
ENV PORT=8080
EXPOSE 8080

# Command: run uvicorn module app:app (adjust if your app is named differently)
WORKDIR /app/backend
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

