import numpy as np
import faiss

def build_index(embeddings):
    embedding_dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)
    return index
