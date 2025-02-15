import streamlit as st
import pickle
import os
import pandas as pd

# Define file paths for pickle files
movies_pickle_path = r'd:\vs_code_files\movie recomendation\movies.pkl'
similarity_pickle_path = r'd:\vs_code_files\movie recomendation\similarity.pkl'




# Check if pickle files exist
if not os.path.exists(movies_pickle_path) or not os.path.exists(similarity_pickle_path):
    st.error("Pickled data files not found. Please run 'preprocess.py' to generate 'movies.pkl' and 'similarity.pkl'.")
else:
    # Load pickled data
    new_df = pickle.load(open(movies_pickle_path, 'rb'))
    similarity = pickle.load(open(similarity_pickle_path, 'rb'))

    # Recommendation function
    def recommend(movie):
        # Normalize the input movie title
        movie_norm = movie.replace(" ", "").lower()
        
        # Find matching row(s) using the precomputed normalized titles
        matching_rows = new_df[new_df['title_norm'] == movie_norm]
        if matching_rows.empty:
            return []
        
        movie_index = matching_rows.index[0]
        # Compute similarity distances
        distances = sorted(
            list(enumerate(similarity[movie_index])),
            reverse=True,
            key=lambda x: x[1]
        )
        # Return the top 5 recommendations (excluding the input movie itself)
        recommendations = [new_df.iloc[i[0]].title for i in distances[1:6]]
        return recommendations

    # Streamlit interface
    st.title("Movie Recommendation System")
    movie_input = st.text_input("Enter a movie title:", "")
    
    if st.button("Get Recommendations"):
        if movie_input:
            recs = recommend(movie_input)
            if recs:
                st.subheader("Recommendations:")
                for rec in recs:
                    st.write(rec)
            else:
                st.error(f"No movie found with title: {movie_input}")
        else:
            st.error("Please enter a movie title.")
