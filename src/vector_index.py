import faiss
import numpy as np
from typing import List, Any, Literal


def build_index(embeddings: List[List[float]], metric: Literal["l2", "cosine"] = "l2") -> Any:
    """
    Build a vector index from a list of embedding vectors using FAISS.

    Args:
        embeddings: A list of embedding vectors, where each vector is a list of floats
        metric: Distance metric to use, either "l2" or "cosine"

    Returns:
        A FAISS index containing the embeddings for efficient similarity search
    """
    if not embeddings:
        raise ValueError("Cannot build index with empty embeddings list")
        
    embedding_dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype("float32")
    
    if metric == "l2":
        index = faiss.IndexFlatL2(embedding_dim)
    else:  # metric == "cosine" due to Literal type constraint
        index = faiss.IndexFlatIP(embedding_dim)
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_np)
        
    index.add(embeddings_np)
    return index
