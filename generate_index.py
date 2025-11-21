# generate_index.py
import sys
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import zipfile
import os

def main(csv_path):
    df = pd.read_csv(csv_path)
    # detect sensible text column
    for col in ['text','utterance','query','sentence','content','prompt']:
        if col in df.columns:
            text_col = col
            break
    else:
        text_col = df.columns[0]

    texts = df[text_col].astype(str).tolist()
    print(f"Using column '{text_col}' with {len(texts)} rows")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True)

    out_dir = os.getcwd()
    emb_path = os.path.join(out_dir, 'embeddings.npy')
    meta_path = os.path.join(out_dir, 'metadata.joblib')
    zip_path = os.path.join(out_dir, 'index_files.zip')

    np.save(emb_path, embeddings)
    joblib.dump({'texts': texts, 'columns': list(df.columns)}, meta_path)

    with zipfile.ZipFile(zip_path, 'w') as z:
        z.write(emb_path, arcname='embeddings.npy')
        z.write(meta_path, arcname='metadata.joblib')

    print("Saved:", emb_path, meta_path, zip_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_index.py path/to/intent.csv")
    else:
        main(sys.argv[1])
