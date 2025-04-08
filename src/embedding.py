from openai import OpenAI
from typing import List, Optional


def get_embedding(text: str, client: Optional[OpenAI] = None, model: str = "text-embedding-3-small") -> List[float]:
    """
    Generate an embedding vector for the provided text using OpenAI's embedding model.

    Args:
        text: The text to generate an embedding for
        client: An optional OpenAI client instance. If not provided, assumes the client is initialized elsewhere
        model: The embedding model to use (default: text-embedding-3-large). 
               (can be changed to 'text-embedding-3-small' for faster operations and reduced costs)

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
