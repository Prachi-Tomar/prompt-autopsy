from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_model_cache = {}

def get_model(model_name: str):
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]

def embed_texts(texts, model_name: str):
    model = get_model(model_name)
    return model.encode(texts, normalize_embeddings=True).tolist()

def pairwise_similarity(labels, vectors):
    mat = cosine_similarity(np.array(vectors))
    out = {}
    for i, a in enumerate(labels):
        out[a] = {}
        for j, b in enumerate(labels):
            out[a][b] = float(mat[i, j])
    return out