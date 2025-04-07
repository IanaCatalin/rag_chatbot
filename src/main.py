from pdf_extraction import extract_text_from_pdf
from text_chunking import chunk_text
from embedding import get_embedding
from vector_index import build_index
from query_handler import query_rag, generate_answer
import glob

def main():
    # 1. Extract text from the PDF
    text = extract_text_from_pdf(pattern="data/*.pdf")

    # 2. Chunk the text
    chunks = chunk_text(text)

    # 3. Generate embeddings for each chunk
    embeddings = [get_embedding(chunk) for chunk in chunks]

    # 4. Build the vector index
    index = build_index(embeddings)

    # 5. Handle a sample query
    query = "What is the main idea discussed in the document?"
    relevant_chunks = query_rag(query, index, chunks)
    answer = generate_answer(query, relevant_chunks)

    print("Answer:", answer)

if __name__ == "__main__":
    main()
