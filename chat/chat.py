import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import textwrap

# -------------------
# Load models
# -------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
    return embedder, qa_model

embedder, qa_model = load_models()

# -------------------
# Functions
# -------------------
def chunk_text(text, chunk_size=200):
    """Split text into smaller chunks."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def build_faiss_index(chunks):
    """Create FAISS index for text chunks."""
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

def retrieve_relevant_chunks(question, chunks, index, top_k=2):
    """Retrieve most relevant chunks using FAISS."""
    q_emb = embedder.encode([question]).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]]

def answer_question(question, context):
    """Use Flan-T5 to generate answer from context + question."""
    prompt = f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    result = qa_model(prompt, max_length=200, do_sample=False)
    return result[0]['generated_text']

# -------------------
# Streamlit UI
# -------------------
st.title("📘 Chat with Your Text")

text_input = st.text_area("Paste your text here:", height=200)
question = st.text_input("Ask a question about the text:")

if st.button("Get Answer"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        # Chunk + Index
        chunks = chunk_text(text_input)
        index, _ = build_faiss_index(chunks)

        # Retrieve relevant parts
        relevant_chunks = retrieve_relevant_chunks(question, chunks, index)

        # Merge them into context
        context = " ".join(relevant_chunks)

        # Get answer
        answer = answer_question(question, context)

        st.subheader("Answer:")
        st.write(textwrap.fill(answer, width=80))
