import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from vector_index import build_index
from embedding import get_embedding

load_dotenv(override=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def query_rag(query, index, chunks, k=3):
    query_embedding = np.array(get_embedding(query)).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return relevant_chunks


def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = (
        f"Based on the following context, answer the question:\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    )
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=150)
    answer = response.choices[0].message.content.strip()
    return answer


