import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import nltk
import gradio as gr

nltk.download('punkt')
ps = PorterStemmer()

# Helper functions
def stem_text(text):
    return " ".join([ps.stem(word) for word in text.split()])

def extract_names(text):
    try:
        return [item['name'].replace(" ", "") for item in ast.literal_eval(text)]
    except:
        return []

# Load & process
movies = pd.read_csv("movies.csv")
credits = pd.read_csv("credits.csv")
df = movies.merge(credits, on="title")
df = df[['movie_id', 'title', 'genres', 'keywords', 'cast', 'crew']]

df['genres'] = df['genres'].apply(extract_names)
df['keywords'] = df['keywords'].apply(extract_names)
df['cast'] = df['cast'].apply(extract_names)
df['crew'] = df['crew'].apply(extract_names)

df['tags'] = df['genres'] + df['keywords'] + df['cast'] + df['crew']
df['tags'] = df['tags'].apply(lambda x: " ".join(x).lower())
df['tags'] = df['tags'].apply(stem_text)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vectors)

movie_titles = sorted(df['title'].tolist())

# Recommendation logic
def recommend(movie):
    movie = movie.lower()
    if movie not in df['title'].str.lower().values:
        return "Movie not found."
    index = df[df['title'].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[index]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return "\n".join([df.iloc[i[0]].title for i in sorted_movies])

# Gradio UI
interface = gr.Interface(
    fn=recommend,
    inputs=gr.Dropdown(choices=movie_titles, label="Select a Movie", allow_custom_value=True, interactive=True),
    outputs="text",
    title="🎬 Movie Recommender",
    description="Select or type a movie name to get 5 similar recommendations"
)

interface.launch()
