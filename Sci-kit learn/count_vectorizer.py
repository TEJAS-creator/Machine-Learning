from sklearn.feature_extraction.text import CountVectorizer

# Sample data
corpus = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love coding in Python"
]

# Initialize CountVectorizer
vectorizer = CountVectorizer()

# Fit and transform
X = vectorizer.fit_transform(corpus)

# Convert to array
print(X.toarray())
print(vectorizer.get_feature_names_out())
