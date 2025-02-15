import numpy as np 
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from nltk.stem.porter import PorterStemmer

# Read data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on="title")

# Keep only the necessary columns
movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]

# Drop rows with null values
movies.dropna(inplace=True)

# Function to convert stringified list to Python list for genres and keywords
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Function to fetch director from crew column
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break  
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Tokenize the overview text
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces in multi-word entries in genres, keywords, cast, and crew
movies["genres"] = movies["genres"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["keywords"] = movies["keywords"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["cast"] = movies["cast"].apply(lambda x: [i.replace(" ", "") for i in x])
movies["crew"] = movies["crew"].apply(lambda x: [i.replace(" ", "") for i in x])

# Combine features into one 'tags' column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create new DataFrame with required columns (explicit copy)
new_df = movies[["movie_id", "title", "tags"]].copy()

# Join list of tags into one string and convert to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Initialize the PorterStemmer and define a stemming function
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

# Apply stemming to tags
new_df['tags'] = new_df['tags'].apply(stem)

# Precompute normalized title (remove spaces and lowercase) for matching
new_df['title_norm'] = new_df['title'].str.replace(" ", "").str.lower()

# Vectorize the tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()

# Compute the cosine similarity matrix
similarity = cosine_similarity(vector)

# Save the preprocessed DataFrame and similarity matrix as pickle files
with open('movies.pkl', 'wb') as f:
    pickle.dump(new_df, f)
with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

print("Preprocessing complete. 'movies.pkl' and 'similarity.pkl' have been generated.")
