import openai
import os


def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

