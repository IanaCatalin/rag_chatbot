import streamlit as st
import glob
import numpy as np
import openai
import os

from pdf_extraction import extract_text_from_pdf
from text_chunking import chunk_text
from embedding import get_embedding
from vector_index import build_index
from query_handler import query_rag, generate_answer

# Set OpenAI API key from environment variable

@st.cache_resource
def load_index_and_chunks(pdf_pattern="data/*.pdf"):
    st.info("Processing PDF(s) and building index... (This may take a minute)")

    # Expand the wildcard pattern to find PDF files
    pdf_files = glob.glob(pdf_pattern)
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found matching the pattern: {pdf_pattern}")

    # Extract text from all PDF files
    all_text = ""
    for pdf_file in pdf_files:
        all_text += extract_text_from_pdf(pdf_file)

    # Split the combined text into chunks
    chunks = chunk_text(all_text)

    # Generate embeddings for each chunk
    embeddings = [get_embedding(chunk) for chunk in chunks]

    # Build the FAISS vector index
    index = build_index(embeddings)

    return index, chunks

# Load (or cache) the index and text chunks.
index, chunks = load_index_and_chunks()

# Streamlit UI
st.title("RAG Chatbot")
st.write("Ask a question about the PDF document(s).")

user_query = st.text_input("Your Question:")

if user_query:
    st.info("Retrieving relevant context and generating answer...")
    # Retrieve relevant text chunks based on the query
    relevant_chunks = query_rag(user_query, index, chunks)
    # Generate an answer using GPT
    answer = generate_answer(user_query, relevant_chunks)
    st.markdown("### Answer")
    st.write(answer)
