from openai import OpenAI
from typing import List, Optional, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm


def get_embedding(text: str, client: Optional[OpenAI] = None, model: str = "text-embedding-3-small") -> List[float]:
    """
    Generate an embedding vector for the provided text using OpenAI's embedding model.

    Args:
        text: The text to generate an embedding for
        client: An optional OpenAI client instance. If not provided, assumes the client is initialized elsewhere
        model: The embedding model to use (default: text-embedding-3-small)

    Returns:
        A list of floats representing the embedding vector
    """
    if client is None:
        raise ValueError("OpenAI client must be provided")

    response = client.embeddings.create(
        input=text,
        model=model
    )

    return response.data[0].embedding


async def get_embeddings_concurrent(
    texts: List[str], 
    client: Optional[OpenAI] = None, 
    model: str = "text-embedding-3-small", 
    batch_size: int = 50,
    show_progress: bool = False
) -> List[List[float]]:
    """
    Generate embedding vectors for multiple texts concurrently using OpenAI's embedding model.

    Args:
        texts: List of texts to generate embeddings for
        client: An optional OpenAI client instance. If not provided, assumes the client is initialized elsewhere
        model: The embedding model to use (default: text-embedding-3-small)
        batch_size: Maximum number of concurrent API calls (default: 50)
        show_progress: Whether to display a progress bar (default: False)

    Returns:
        A list of embedding vectors, where each vector is a list of floats
    """
    if client is None:
        raise ValueError("OpenAI client must be provided")
    
    # Create a wrapper function for concurrent execution
    def get_single_embedding(text):
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    
    # Process in batches to limit concurrency
    embeddings = []
    
    # Setup progress bar if requested
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating embeddings", total=len(range(0, len(texts), batch_size)))
    
    for i in iterator:
        batch_texts = texts[i:i+batch_size]
        
        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            batch_embeddings = list(executor.map(get_single_embedding, batch_texts))
            
        embeddings.extend(batch_embeddings)
    
    return embeddings
