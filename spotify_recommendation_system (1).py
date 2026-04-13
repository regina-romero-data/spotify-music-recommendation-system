"""
=============================================================================
Spotify Music Recommendation System — Content-Based Filtering
=============================================================================
Project:    Academic Project — Data Mining
Author:     Regina Romero de León
Year:       2025

Description:
    Content-based music recommendation system built on the Spotify Top 2000s
    dataset (1956–2019). Uses TF-IDF vectorization and cosine similarity to
    find songs that share similar audio characteristics — genre, artist,
    energy, danceability, and valence — and recommend them based on a given
    input track.

Methodology:
    - Data Loading & Inspection
    - Feature Engineering (combined text representation)
    - TF-IDF Vectorization
    - Cosine Similarity Matrix
    - Content-Based Recommendation Function

Dataset:
    Spotify Top 2000s Megadataset — Kaggle
    https://www.kaggle.com/datasets/iamsumat/spotify-top-2000s-megadataset

Tools: Python · Pandas · Scikit-learn · TF-IDF · Cosine Similarity · NLP
=============================================================================
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# =============================================================================
# 2. LOAD DATA
# =============================================================================

df = pd.read_csv('Spotify-2000.csv')

# Display first rows to verify structure
df.head(40)

# =============================================================================
# 3. DATA INSPECTION & CLEANING
# =============================================================================

# Inspect key columns that will be used to build song features
# We use genre, artist, and "vibe" measures (energy, danceability, valence)
print(df[['Top Genre', 'Artist', 'Energy', 'Danceability', 'Valence']].head())

# Fill null values and clean whitespace in text columns
df['Top Genre'] = df['Top Genre'].fillna('').str.strip()
df['Artist'] = df['Artist'].fillna('').str.strip()

# Fill null values in key numerical columns
df['Energy'] = df['Energy'].fillna(0)
df['Danceability'] = df['Danceability'].fillna(0)
df['Valence'] = df['Valence'].fillna(0)

# =============================================================================
# 4. FEATURE ENGINEERING — COMBINED TEXT REPRESENTATION
# =============================================================================

# Create a new column that combines key song information into a single text string.
# This 'combined_features' column merges:
#   - Top Genre, Artist (text columns — commas replaced with spaces to avoid
#     TF-IDF treating them as separate tokens)
#   - Energy, Danceability, Valence (numerical columns converted to string)
# This unified representation allows the TF-IDF vectorizer to process each
# song as a "document" and measure similarity between songs.

df['combined_features'] = (
    df['Top Genre'].str.replace(',', ' ', regex=False) + ' ' +
    df['Artist'].str.replace(',', ' ', regex=False) + ' ' +
    df['Energy'].astype(str) + ' ' +
    df['Danceability'].astype(str) + ' ' +
    df['Valence'].astype(str)
).str.strip()

# Remove rows where the combined feature string is empty
df = df[df['combined_features'] != '']

# =============================================================================
# 5. TF-IDF VECTORIZATION
# =============================================================================

# Transform combined text into numerical vectors using TF-IDF.
# TF-IDF (Term Frequency - Inverse Document Frequency) measures the relevance
# of each word within a song relative to the entire dataset.
# The result is a matrix where each row = a song, each column = a unique term.

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_features'])

# Display resulting matrix dimensions (songs x unique terms)
print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

# =============================================================================
# 6. COSINE SIMILARITY MATRIX
# =============================================================================

# Compute cosine similarity between all songs based on the TF-IDF matrix.
# The result is a square matrix where cell (i, j) represents
# how similar song i is to song j.
# This matrix is the foundation for generating recommendations.

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# =============================================================================
# 7. SONG INDEX MAPPING
# =============================================================================

# Create a song index that maps each song to its position in the DataFrame.
# We combine 'Title' and 'Artist' as a unique key to avoid confusion
# between songs with the same title by different artists.
# This allows us to quickly look up a song's position when generating
# recommendations.

df['song_key'] = (df['Title'].astype(str) + ' - ' + df['Artist'].astype(str)).str.strip()

indices = pd.Series(df.index, index=df['song_key']).drop_duplicates()

# =============================================================================
# 8. RECOMMENDATION FUNCTION
# =============================================================================

def get_recommendations(song_name, num_recommendations=5):
    """
    Returns the top N most similar songs to the given input song.

    Parameters:
        song_name (str): Song name in the format "Title - Artist"
        num_recommendations (int): Number of recommendations to return

    Returns:
        DataFrame with Title and Artist of recommended songs,
        or an error message if the song is not found in the dataset.

    How it works:
        1. Looks up the song's index in the DataFrame using the 'indices' dict.
        2. Retrieves the cosine similarity scores for that song vs. all others.
        3. Sorts songs from most to least similar.
        4. Excludes the input song itself.
        5. Returns the top N most similar songs.
    """
    if song_name not in indices:
        return f"'{song_name}' was not found in the dataset."

    # Get the song's index in the DataFrame
    idx = indices[song_name]

    # Get similarity scores for this song vs. all others
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort songs by similarity score (descending)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclude the input song itself and select top N
    sim_scores = sim_scores[1:num_recommendations + 1]

    # Get indices of recommended songs
    song_indices = [i[0] for i in sim_scores]

    # Return title and artist of recommended songs
    return df[['Title', 'Artist']].iloc[song_indices]

# =============================================================================
# 9. MODEL VALIDATION — TEST CASES
# =============================================================================

# Test the recommendation function with real examples.
# This demonstrates that the system identifies songs with similar characteristics
# using only the text information processed through TF-IDF.

print("Recommended songs similar to 'Clint Eastwood - Gorillaz':")
print(get_recommendations("Clint Eastwood - Gorillaz", 20))

print("\nRecommended songs similar to 'Counting Stars - OneRepublic':")
print(get_recommendations("Counting Stars - OneRepublic", 20))
