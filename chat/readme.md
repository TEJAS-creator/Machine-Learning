# 📘 Chat with Your Text

This is a simple **Streamlit app** that lets you **paste any text and ask questions about it**.  
It uses:

- [SentenceTransformers](https://www.sbert.net/) → to create embeddings of text  
- [FAISS](https://faiss.ai/) → to search relevant parts of the text  
- [Flan-T5](https://huggingface.co/google/flan-t5-base) → to generate answers  
- [Streamlit](https://streamlit.io/) → for the interactive web app  

---

## 🚀 Features
- Paste any document or notes into the app  
- Ask natural language questions about the text  
- The app finds **relevant chunks** of your text using FAISS similarity search  
- Flan-T5 generates a clean answer based on retrieved context  

---

## 🛠️ Workflow (Step by Step)

Here’s what happens inside the app:

Models load once (cached):

SentenceTransformer → creates embeddings (vectors) for text

Flan-T5 → answers questions

User interface appears:

A box to paste your text

A box to type your question

A button to get the answer

When you click "Get Answer":

The text is split into chunks (default: 200 words per chunk).

Each chunk is converted into embeddings (vectors of numbers).

All chunk vectors are stored in a FAISS index (fast search).

Question processing:

The question is also converted into an embedding.

FAISS finds the most similar chunks to the question.

Context building:

Retrieved chunks are combined into a single context string.

Answer generation:

The context + question is sent to Flan-T5.

Model generates a natural language answer.

Display:

The answer is shown in the Streamlit app.

---

## 🔁 Example

### Input text:
```
Python is a high-level programming language. 
It is widely used for web development, data science, 
machine learning, and automation.
```

### Question:
```
What is Python used for?
```

### Output:
```
Python is used for web development, data science, machine learning, and automation.
```

---


## 📌 Notes
- **Chunking**: Keeps documents manageable and searchable.  
- **Embeddings + FAISS**: Finds semantically relevant parts (not just keyword matches).  
- **Flan-T5**: Converts context into a human-friendly answer.  
- **Streamlit**: Makes everything interactive with minimal code.  

---

# Sentence Embeddings with all-MiniLM-L6-v2

## 1. Why do we need to convert text into numbers?
Computers cannot directly "understand" natural language like humans.  
They only work with numbers (vectors, matrices, tensors).  
To enable machines to process text meaningfully, we must **represent sentences as numbers**.

---

## 2. What are embeddings?
Embeddings are **dense numerical vectors** that capture the *semantic meaning* of text.  

- Example:

These vectors are **close** because the sentences mean almost the same thing.

- But:
---

## 3. Why embeddings are useful
Embeddings allow applications to **understand semantic similarity** rather than just keywords.

- **Semantic Search** 🔍  
Query: `"How to jog?"`  
Retrieved text: `"Tips for running"`  
(Even though the word "jog" ≠ "run", embeddings recognize the similarity.)

- **Clustering & Similarity** 🤝  
Sentences with related meanings form clusters in vector space.

- **Retrieval before Question Answering** 🧠  
1. Store document embeddings in a vector database (e.g., FAISS).  
2. Convert user query into an embedding.  
3. Find closest vectors (semantically related text).  
4. Pass retrieved text to a QA model.  

Without embeddings, systems rely on **keyword matching**, which fails with synonyms or paraphrasing.

---

## 4. Model used: `all-MiniLM-L6-v2`
- Lightweight & fast transformer-based sentence embedding model.  
- Produces **384-dimensional vectors**.  
- Great trade-off between speed and accuracy.

---

## 5. Example Code (Tiny Demo)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Sentences
s1 = "I like apples"
s2 = "I enjoy eating fruits"
s3 = "I drive a car"

# Convert to embeddings
emb1 = model.encode(s1)
emb2 = model.encode(s2)
emb3 = model.encode(s3)

# Print embeddings (first 5 numbers for readability)
print(f'"{s1}" → {np.round(emb1[:5], 2)} ...')
print(f'"{s2}" → {np.round(emb2[:5], 2)} ...')
print(f'"{s3}" → {np.round(emb3[:5], 2)} ...')

# Compare similarity
print("s1 vs s2 similarity:", util.cos_sim(emb1, emb2).item())  # High similarity
print("s1 vs s3 similarity:", util.cos_sim(emb1, emb3).item())  # Low similarity

"I like apples" → [ 0.12 -0.45  0.87  0.33 -0.21] ...
"I enjoy eating fruits" → [ 0.10 -0.43  0.85  0.31 -0.20] ...
"I drive a car" → [-0.72  0.33 -0.90  0.15  0.48] ...

s1 vs s2 similarity: 0.89
s1 vs s3 similarity: 0.15
