import numpy as np
from openai import OpenAI
from embedding import get_embedding
from typing import List, Any, Optional


def query_rag(query: str,
              index: Any,
              chunks: List[str],
              k: int = 3,
              client: Optional[OpenAI] = None) -> List[str]:
    """
    Retrieve relevant text chunks based on a query using vector similarity search.

    Args:
        query: The user's question or query
        index: A FAISS index containing embeddings of text chunks
        chunks: A list of text chunks corresponding to the embeddings in the index
        k: Number of relevant chunks to retrieve (default: 3)
        client: An optional OpenAI client instance. If not provided, assumes the client is initialized elsewhere

    Returns:
        A list of text chunks most relevant to the query
    """

    if client is None:
        raise ValueError("OpenAI client must be provided")

    query_embedding = np.array(get_embedding(
        query, client=client)).astype("float32").reshape(1, -1)
    _, indices = index.search(query_embedding, k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks


def generate_answer(query: str, context_chunks: List[str],
                    client: Optional[OpenAI] = None,
                    model: str = "gpt-4o",
                    temperature: float = 0.7) -> str:
    """
    Generate an answer to the query based on the provided context chunks using OpenAI's API.

    Args:
        query: The user's question or query
        context_chunks: A list of text chunks providing context for answering the query
        client: An optional OpenAI client instance. If not provided, assumes the client is initialized elsewhere
        model: The model to use for generating the answer (default: gpt-4o)
        temperature: Controls randomness in the response (default: 0.7)

    Returns:
        A string containing the generated answer
    """
    if client is None:
        raise ValueError("OpenAI client must be provided")

    # Read the system prompt from file
    system_prompt = ""
    try:
        with open("prompts/system.prompt", "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception as e:
        print(f"Error reading system prompt: {e}")
        system_prompt = "Ești un asistent virtual inteligent creat de E.ON România."
    
    # Join context chunks and create the conversation
    context = "\n\n".join(context_chunks)
    
    # Create messages with system prompt and user query with context
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nÎntrebare: {query}"}
    ]
    
    # Generate the response
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    
    answer = response.choices[0].message.content.strip()
    return answer
