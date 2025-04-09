import os
import glob
import asyncio
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

from data_extraction import extract_text_from_pdf, extract_from_json
from text_chunking import chunk_text
from embedding import get_embeddings_concurrent
from vector_index import build_index
from query_handler import query_rag, generate_answer

# Load environment variables
load_dotenv(override=True)

# Set page configuration
st.set_page_config(
    page_title="E.ON - Ioana Doi",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066B3;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #444;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #0066B3;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton button:hover {
        background-color: #004C8C;
    }
    .info-box {
        background-color: #E6F2FF;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #0066B3;
        margin-bottom: 1rem;
    }
    .answer-container {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #DDD;
        margin-top: 1rem;
    }
    .source-container {
        background-color: #F0F0F0;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #0066B3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@st.cache_resource
def load_index_and_chunks(pdf_pattern="data/*.pdf", json_path="json_data/eon_data.json"):
    with st.spinner("Se procesează documentele și se construiește indexul de cunoștințe..."):
        # Extrage text din fișierele PDF
        pdf_files = glob.glob(pdf_pattern)
        if not pdf_files:
            st.warning(
                f"Nu s-au găsit fișiere PDF care să corespundă modelului: {pdf_pattern}")
            pdf_text = ""
        else:
            pdf_text = extract_text_from_pdf(pdf_pattern)
            st.success(f"✅ S-au extras {len(pdf_text):,} caractere din PDF-uri")

        # Extrage text din fișierele JSON
        try:
            json_data = extract_from_json(json_path, include_urls=False)

            # Filtrează zgomotul din textul JSON pe baza conținutului din text_1.txt
            filtered_json_data = {}

            # Obține liniile din text_1.txt pentru a le folosi ca filtru
            with open("filter_text/text_1.txt", "r", encoding="utf-8") as f:
                filter_lines = set(line.strip() for line in f if line.strip())

            # Filtrează liniile de zgomot din datele JSON
            for key, value in json_data.items():
                lines = value.split("\n")
                filtered_lines = [line for line in lines if line.strip(
                ) and line.strip() not in filter_lines]
                filtered_json_data[key] = "\n".join(filtered_lines)

            json_text = "\n\n".join(filtered_json_data.values())
            st.success(
                f"✅ S-au extras {len(json_text):,} caractere din JSON (după filtrare)")
        except Exception as e:
            st.error(f"Eroare la extragerea datelor JSON: {e}")
            json_text = ""

        # Combină textul din ambele surse
        combined_text = pdf_text + "\n\n" + json_text

        # Împarte textul combinat în fragmente
        chunks = chunk_text(combined_text)
        st.success(f"✅ S-au creat {len(chunks):,} fragmente de text")

        # Generează embeddings pentru fiecare fragment folosind procesarea concurentă
        async def get_embeddings():
            return await get_embeddings_concurrent(
                texts=chunks,
                client=client,
                batch_size=200,
                show_progress=True
            )

        # Rulează funcția asincronă pentru a obține embeddings
        embeddings = asyncio.run(get_embeddings())
        st.success(f"✅ S-au generat {len(embeddings):,} embeddings")

        # Construiește indexul vectorial FAISS
        index = build_index(embeddings, metric='l2')
        st.success("✅ Indexul vectorial a fost construit cu succes")

        return index, chunks


# Sidebar with information
with st.sidebar:
    st.image("https://www.ifacts.se/wp-content/uploads/2017/12/ifacts__0001_e.on_.png.png", width=150)
    st.markdown("### Despre")
    st.info(
        "Acest asistent AI te ajută să găsești informații despre serviciile, produsele și procedurile E.ON. "
        "Pune orice întrebare legată de ofertele E.ON România."
    )

    st.markdown("### Exemple")
    st.markdown("""
    - Cum pot instala panouri solare?
    - Care sunt beneficiile E.ON Solar?
    - Cum îmi creez un cont E.ON Myline?
    - Ce metode de plată sunt disponibile?
    """)

    st.markdown("---")
    st.markdown("### Surse de date")
    st.markdown("Acest asistent este alimentat de:")
    st.markdown("- Conținutul site-ului web E.ON")
    st.markdown("- Documentația in format PDF")

# Main content
st.markdown('<h1 class="main-header">E.ON -- Ioana Doi!</h1>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ghidul tău virtual pentru serviciile și produsele E.ON</p>',
            unsafe_allow_html=True)

# Load (or cache) the index and text chunks.
index, chunks = load_index_and_chunks()

# Create two columns for the query input
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input(
        "", placeholder="Întreabă-mă orice despre serviciile E.ON...", label_visibility="collapsed")
with col2:
    search_button = st.button("Întreabă", use_container_width=True)

# Process the query
if user_query and (search_button or 'last_query' not in st.session_state or st.session_state.last_query != user_query):
    st.session_state.last_query = user_query

    with st.spinner("Caut cel mai bun răspuns..."):
        # Retrieve relevant text chunks based on the query
        relevant_chunks = query_rag(
            user_query, index, chunks, k=7, client=client)

        # Generate an answer using GPT
        answer = generate_answer(user_query, relevant_chunks, client=client)

        # Display the answer
        st.markdown('<div class="answer-container">', unsafe_allow_html=True)
        st.markdown("### Răspuns")
        st.markdown(answer)
        st.markdown('</div>', unsafe_allow_html=True)

        # Show relevant chunks if requested
        with st.expander("Vizualizează informațiile sursă"):
            st.markdown("### Context Relevant")
            for i, chunk in enumerate(relevant_chunks):
                st.markdown(f"**Sursa {i+1}**")
                st.markdown(
                    f'<div class="source-container">{chunk}</div>', unsafe_allow_html=True)
