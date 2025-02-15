# Movie Recommendation System

## Overview
This project is a content-based movie recommendation system using Natural Language Processing (NLP) and machine learning techniques. The system processes movie metadata, computes similarities between movies, and provides personalized movie recommendations.

## Features
- Extracts and preprocesses movie data from CSV files
- Uses NLP techniques like stemming to refine text-based features
- Computes cosine similarity between movie descriptions
- Provides movie recommendations based on user input
- Implements a Streamlit web app for easy interaction

## Installation
### Prerequisites
Ensure you have Python installed along with the necessary dependencies:
```bash
pip install numpy pandas scikit-learn nltk streamlit
```

### Clone the Repository
```bash
git clone https://github.com/yourusername/movie-recommendation.git
cd movie-recommendation
```

## Data Preprocessing
Before running the recommendation system, you need to preprocess the data:
```bash
python preprocess.py
```
This script will:
- Load and clean the movie dataset
- Extract relevant features
- Compute cosine similarity
- Save the processed data as pickle files (`movies.pkl` and `similarity.pkl`)

## Running the Web App
After preprocessing, start the Streamlit application:
```bash
streamlit run app.py
```
This will launch the recommendation system in your browser.

## Usage
1. Enter a movie title in the input box.
2. Click the "Get Recommendations" button.
3. The system will return five recommended movies similar to the input.

## File Structure
```
movie-recommendation/
│── preprocess.py    # Preprocesses data and saves it as pickle files
│── app.py           # Streamlit app for movie recommendations
│── movies.pkl       # Preprocessed movie data
│── similarity.pkl   # Cosine similarity matrix
│── tmdb_5000_movies.csv  # Movie dataset
│── tmdb_5000_credits.csv # Movie credits dataset
│── README.md        # Project documentation
```

## Potential Enhancements
- Integrate a more advanced NLP model for better recommendations
- Add more metadata (e.g., user ratings) to improve accuracy
- Implement a collaborative filtering approach

## License
This project is open-source and available under the MIT License.

## Contributors
- Royson Salis ([GitHub Profile](https://github.com/Royson-salis-18))

Feel free to contribute, report issues, or suggest improvements!

