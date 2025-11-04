"""
Robust FastAPI backend for an intents-based ML assistant.

Features:
- Loads intents from backend/intents.csv (columns: 'intent', 'text') or via /upload_intents (CSV multipart).
- Builds embeddings using SentenceTransformer.
- Uses FAISS if available (fast). If faiss is missing, falls back to scikit-learn NearestNeighbors (works on Windows).
- Endpoints:
    GET  /health
    GET  /metadata
    POST /predict          -> JSON {"query":"...", "top_k":3}
    POST /upload_intents   -> multipart file (CSV)
- Persists index/embeddings & metadata to joblib files for faster restarts.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
import joblib
import pandas as pd
import numpy as np

# lazy import for heavy libs
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

# Try to import faiss; if not present, we'll fallback to sklearn
HAS_FAISS = True
try:
    import faiss
except Exception:
    HAS_FAISS = False

# sklearn fallback
from sklearn.neighbors import NearestNeighbors

# ----------------------
# Config & paths
# ----------------------
INTENTS_CSV = "intents.csv"
INDEX_FILE = "faiss_index.joblib"         # used only if faiss present
EMBEDDINGS_FILE = "embeddings.npy"       # fallback storage for sklearn or general use
METADATA_FILE = "metadata.joblib"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # this model's embedding dim

# ----------------------
# FastAPI app
# ----------------------
app = FastAPI(title="ML Assistant Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Request/Response models
# ----------------------
class PredictRequest(BaseModel):
    query: str
    top_k: int = 3

class PredictResult(BaseModel):
    intent: str
    text: str
    score: float

# ----------------------
# Globals
# ----------------------
model = None
faiss_index = None
sk_index = None  # sklearn NearestNeighbors fallback
embeddings = None  # numpy array (N x D)
metadata: List[Dict] = []

# ----------------------
# Helpers
# ----------------------
def ensure_model_loaded():
    global model
    if model is None:
        if SentenceTransformer is None:
            raise HTTPException(status_code=500, detail="sentence-transformers not installed.")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)

def try_load_persisted():
    """
    Load persisted data if present:
      - If faiss is available and INDEX_FILE exists, try loading it.
      - Load embeddings.npy and metadata.joblib if present (sklearn fallback).
    """
    global faiss_index, embeddings, metadata, sk_index

    if HAS_FAISS and os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        try:
            faiss_index = joblib.load(INDEX_FILE)
            metadata = joblib.load(METADATA_FILE)
            return True
        except Exception:
            pass

    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(METADATA_FILE):
        try:
            embeddings = np.load(EMBEDDINGS_FILE)
            metadata = joblib.load(METADATA_FILE)
            # build sklearn index
            if embeddings is not None and len(embeddings) > 0:
                sk_index = NearestNeighbors(n_neighbors=5, metric="cosine")
                sk_index.fit(embeddings)
            return True
        except Exception:
            pass

    return False

def build_index_from_dataframe(df: pd.DataFrame):
    """
    Build index and persist it. Expects 'intent' and 'text' columns.
    """
    global faiss_index, embeddings, metadata, sk_index
    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("CSV must contain 'intent' and 'text' columns")

    texts = df["text"].astype(str).tolist()
    intents = df["intent"].astype(str).tolist()

    ensure_model_loaded()
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)

    metadata = [{"intent": intents[i], "text": texts[i]} for i in range(len(texts))]

    # Persist embeddings & metadata
    np.save(EMBEDDINGS_FILE, embs)
    joblib.dump(metadata, METADATA_FILE)

    # Build index
    if HAS_FAISS:
        # normalize for inner product (cosine-like)
        faiss.normalize_L2(embs)
        d = embs.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embs)
        faiss_index = index
        # persist faiss index using joblib
        joblib.dump(faiss_index, INDEX_FILE)
        # keep embeddings var too
        embeddings = embs
        sk_index = None
    else:
        # sklearn fallback: NearestNeighbors with cosine distance
        sk = NearestNeighbors(n_neighbors=min(10, len(embs)), metric="cosine")
        sk.fit(embs)
        sk_index = sk
        embeddings = embs
        faiss_index = None

def ensure_index():
    """
    Ensure an index (faiss or sklearn) is ready. Tries to load persisted index, else builds from CSV.
    """
    global faiss_index, sk_index, metadata, embeddings
    if faiss_index is not None or sk_index is not None:
        return
    if try_load_persisted():
        return
    if os.path.exists(INTENTS_CSV):
        df = pd.read_csv(INTENTS_CSV)
        build_index_from_dataframe(df)

def search(query: str, top_k: int = 3) -> List[PredictResult]:
    """
    Search the index and return PredictResult list.
    Scores are cosine-similarity-like (higher = better).
    """
    ensure_model_loaded()
    ensure_index()

    if faiss_index is not None:
        q_emb = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = faiss_index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            md = metadata[idx]
            results.append(PredictResult(intent=md["intent"], text=md["text"], score=float(score)))
        return results

    if sk_index is not None and embeddings is not None:
        q_emb = model.encode([query], convert_to_numpy=True)
        # sklearn returns distances (cosine): distance in [0, 2]; similarity = 1 - distance
        distances, indices = sk_index.kneighbors(q_emb, n_neighbors=min(top_k, len(embeddings)))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            score = 1.0 - float(dist)  # convert to similarity-like
            md = metadata[idx]
            results.append(PredictResult(intent=md["intent"], text=md["text"], score=score))
        return results

    raise HTTPException(status_code=500, detail="No index available. Upload intents or place intents.csv in backend/")

# ----------------------
# Startup: try to load index (non-fatal)
# ----------------------
@app.on_event("startup")
def startup_event():
    # do not raise on failure; server should still start
    try:
        ensure_model_loaded()
    except Exception as e:
        print("Warning: model not loaded on startup:", str(e))
    try:
        ensure_index()
    except Exception as e:
        print("Warning: index not built/loaded on startup:", str(e))

# ----------------------
# Routes
# ----------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "faiss_available": HAS_FAISS,
        "index_built": (faiss_index is not None) or (sk_index is not None)
    }

@app.get("/metadata")
def get_metadata():
    return {"count": len(metadata), "samples": metadata[:20]}

@app.post("/predict", response_model=List[PredictResult])
def predict(req: PredictRequest):
    if not req.query or req.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query must be a non-empty string")
    results = search(req.query, top_k=req.top_k)
    return results

@app.post("/upload_intents")
async def upload_intents(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    contents = await file.read()
    from io import StringIO
    try:
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if "intent" not in df.columns or "text" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must have 'intent' and 'text' columns")

    # save CSV and rebuild index
    df.to_csv(INTENTS_CSV, index=False)
    build_index_from_dataframe(df)
    return {"status": "ok", "num_examples": len(df)}

# ----------------------
# Run convenience (local dev)
# ----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
