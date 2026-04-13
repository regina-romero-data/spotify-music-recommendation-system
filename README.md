# spotify-music-recommendation-system
Content-based music recommendation system using TF-IDF and cosine similarity on the Spotify Top 2000s dataset.
Aquí va el README completo listo para copiar y pegar:

markdown# Spotify Music Recommendation System — Content-Based Filtering

## Overview
Content-based music recommendation system built on the Spotify Top 2000s 
dataset (1956–2019). Uses TF-IDF vectorization and cosine similarity to 
recommend songs that share similar audio characteristics based on a given 
input track.

## Business Problem
How can we recommend songs with a similar "vibe" using only audio features 
and metadata — without relying on user listening history or ratings?

## How It Works
1. **Feature Engineering** — Combines genre, artist, energy, danceability, 
   and valence into a single text representation per song.
2. **TF-IDF Vectorization** — Transforms the combined text into numerical 
   vectors measuring the relevance of each term across the dataset.
3. **Cosine Similarity** — Computes a similarity score between every pair 
   of songs in the dataset.
4. **Recommendation Function** — Given a song input, returns the top N 
   most similar songs ranked by similarity score.

## Dataset
- **Source:** Kaggle — Spotify Top 2000s Megadataset
- **Link:** https://www.kaggle.com/datasets/iamsumat/spotify-top-2000s-megadataset
- **Size:** ~2,000 songs · 15 audio features · 1956–2019

## Key Features Used
| Feature | Description |
|---|---|
| Top Genre | Musical genre of the song |
| Artist | Artist name |
| Energy | How energetic the song feels (0–100) |
| Danceability | How easy it is to dance to (0–100) |
| Valence | Positivity of the song's mood (0–100) |

## Test Cases
**Input:** `Clint Eastwood - Gorillaz`
→ Recommends songs sharing similar genre, artist, and energy profile.

**Input:** `Counting Stars - OneRepublic`
→ Recommends upbeat pop songs with matching danceability and mood.

## Tech Stack
- Python · Pandas · Scikit-learn
- TF-IDF · Cosine Similarity · NLP · Feature Engineering

## Files
- `spotify_recommendation_system.py` — Clean Python script
- `Sistema_de_recomendacion_Spotify.html` — Original Jupyter notebook export

## Author
Regina Romero de León · Business Intelligence Student · Tec de Monterrey
