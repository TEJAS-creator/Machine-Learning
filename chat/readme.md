# 📘 Chat with Your Text

An AI-powered **Streamlit application** that lets users upload or paste any text and ask questions about it. The app uses **Sentence Transformers** for embedding, **FAISS** for similarity search, and **Flan-T5** for generating natural language answers.

- [SentenceTransformers](https://www.sbert.net/) → to create embeddings of text  
- [FAISS](https://faiss.ai/) → to search relevant parts of the text  
- [Flan-T5](https://huggingface.co/google/flan-t5-base) → to generate answers  
- [Streamlit](https://streamlit.io/) → for the interactive web app  

---

## 🚀 Features
- 🧠 Understands and answers questions from any text
- ⚡ Fast retrieval using FAISS (vector similarity search)
- 🤖 Uses FLAN-T5 for natural and contextual answers
- 🧩 Text is chunked intelligently for better context understanding
- 💬 Detects unrelated questions using a similarity threshold

---

## 🛠️ Tech Stack
| Component | Purpose |
|------------|----------|
| **Streamlit** | User interface for interaction |
| **Sentence Transformers (all-MiniLM-L6-v2)** | Generates semantic embeddings |
| **FAISS** | Efficient similarity search for text chunks |
| **Flan-T5** | Generates answers from context and question |
| **NumPy** | Array manipulation and vector math |

---

## 📂 How It Works
1. **Input**: User pastes or uploads text.
2. **Chunking**: The text is split into overlapping chunks for better retrieval.
3. **Embedding**: Each chunk is converted into a numerical vector using `SentenceTransformer`.
4. **Indexing**: FAISS stores and searches through these embeddings efficiently.
5. **Retrieval**: The most relevant chunks to the question are found.
6. **Generation**: FLAN-T5 generates the answer using the retrieved context.
7. **Thresholding**: If similarity is too low, it warns that the question is unrelated.

---
