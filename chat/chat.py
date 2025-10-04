import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import textwrap

# -------------------
# Load models efficiently (cache to avoid reloading)
# -------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    qa_model = pipeline("text2text-generation", model="google/flan-t5-base", device=-1)
    return embedder, qa_model

embedder, qa_model = load_models()

# -------------------
# Functions
# -------------------
def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    # Split text into smaller overlapping chunks for better context.
    words = text.split()
    chunks = []
    step = chunk_size // 2  # Overlap chunks for better continuity
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

@st.cache_data(show_spinner=False)
def build_faiss_index(chunks: list[str]):
    """Create FAISS index for text chunks."""
    embeddings = embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Use Inner Product since embeddings are normalized
    index.add(embeddings.astype("float32"))
    return index, embeddings

def retrieve_relevant_chunks(question: str, chunks: list[str], index, top_k: int = 2) -> list[str]:
    """Retrieve most relevant chunks using FAISS."""
    q_emb = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]], float(np.max(scores))

def answer_question(question: str, context: str) -> str:
    """Generate answer from question and context."""
    prompt = f"Answer based on context:\n\nContext: {context}\n\nQuestion: {question}\nAnswer:"
    result = qa_model(prompt, max_length=150, do_sample=False, truncation=True)
    return result[0]['generated_text'].strip()

# -------------------
# Streamlit UI
# -------------------
st.title("📘 Chat with Your Text ")
st.write("Upload or paste your text and ask questions interactively!")

text_input = st.text_area("📄 Paste your text here:", height=200)
question = st.text_input("❓ Ask a question about the text:")

if st.button("🔍 Get Answer"):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text first.")
    elif not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Analyzing and generating response..."):
            # Chunk + Index
            chunks = chunk_text(text_input)
            index, _ = build_faiss_index(chunks)

            # Retrieve relevant parts
            relevant_chunks, max_sim = retrieve_relevant_chunks(question, chunks, index)

            # Relevance threshold
            RELEVANCE_THRESHOLD = 0.35  

            st.subheader("Answer:")
            if max_sim < RELEVANCE_THRESHOLD:
                st.info("🤔 This question seems unrelated to the provided text.")
            else:
                context = " ".join(relevant_chunks)
                answer = answer_question(question, context)
                st.success(textwrap.fill(answer, width=80))
