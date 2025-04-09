import os
from embedding import get_embeddings_concurrent
from text_chunking import chunk_text
from vector_index import build_index
from query_handler import query_rag, generate_answer
from data_extraction import extract_text_from_pdf, extract_from_json

from dotenv import load_dotenv

load_dotenv(override=True)


if __name__ == "__main__":
    from openai import OpenAI

    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # 1. Extract text from both PDF and JSON files
    print("Extracting text from PDF files...")
    pdf_text = extract_text_from_pdf("data/*.pdf")
    print(f"Extracted {len(pdf_text)} characters of text from PDFs")

    print("Extracting text from JSON files...")
    json_data = extract_from_json("json_data/eon_data.json", include_urls=False)


    # Filter out noise from the JSON text based on text_1.txt content
    filtered_json_data = {}
    
    # Get the lines from text_1.txt to use as a filter
    with open("filter_text/text_1.txt", "r", encoding="utf-8") as f:
        filter_lines = set(line.strip() for line in f if line.strip())
    
    # Filter out the noise lines from the JSON data
    for key, value in json_data.items():
        lines = value.split("\n")
        filtered_lines = [line for line in lines if line.strip() and line.strip() not in filter_lines]
        filtered_json_data[key] = "\n".join(filtered_lines)
    
    json_text = "\n\n".join(filtered_json_data.values())
    print(f"Extracted {len(json_text)} characters of text from JSON (after filtering noise)")

    # Combine text from both sources
    combined_text = pdf_text + "\n\n" + json_text
    print(f"Combined text: {len(combined_text)} characters")

    # 2. Chunk the text
    print("Chunking text into manageable segments...")
    chunks = chunk_text(combined_text)
    print(f"Created {len(chunks)} text chunks")

    # 3. Generate embeddings for each chunk
    print("Generating embeddings for each chunk...")
    import asyncio
    
    # Use the concurrent method from embedding.py
    async def get_embeddings():
        return await get_embeddings_concurrent(
            texts=chunks,
            client=client,
            batch_size=200,
            show_progress=True
        )
    
    # Run the async function to get embeddings
    embeddings = asyncio.run(get_embeddings())
    
    print(f"Generated {len(embeddings)} embeddings")

    # 4. Build the vector index
    print("Building vector index...")
    index = build_index(embeddings, metric='l2')
    print("Vector index built successfully")

    # 5. Handle a sample query
    query = "E inseamna E.ON Solar Casa Verde"
    print(f"Processing query: '{query}'")
    print("Retrieving relevant chunks (limited to 10)...")
    relevant_chunks = query_rag(query, index, chunks, k=7, client=client)
    print(f"Found {len(relevant_chunks)} relevant chunks")

    print("\n" + "="*80)
    print("RELEVANT CHUNKS:".center(80))
    print("="*80)
    for i, chunk in enumerate(relevant_chunks):
        print("\n")
        print(f"{'#'*50}".center(80))
        print(f"CHUNK {i+1}:".center(80))
        print(f"{'#'*50}".center(80))
        print("\n")
        print(chunk)
    print("="*80 + "\n")

    print("Generating answer...")
    answer = generate_answer(query, relevant_chunks, client=client)
    print("\n" + "*"*80)
    print("ANSWER:".center(80))
    print("*"*80)
    print(f"\n\t{answer}\n")
    print("*"*80)
