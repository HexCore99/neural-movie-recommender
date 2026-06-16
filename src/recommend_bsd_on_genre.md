# recommend_bsd_on_genre.py Explained

This file defines a `MovieRecommender` class. It loads the trained neural recommendation model, reads movie metadata, fetches optional TMDB details, and recommends movies using a hybrid score:

```text
75% learned movie embedding similarity
25% genre similarity
```

## 1. Imports

```python
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
```

The imports support:

- path handling with `Path`
- JSON poster/detail cache handling
- environment variables for TMDB credentials
- vector math with `numpy`
- CSV/dataframe work with `pandas`
- TMDB HTTP requests with `requests`
- loading the trained Keras model with TensorFlow

## 2. MovieRecommender Class

```python
class MovieRecommender:
```

The whole recommendation system is wrapped in one class. When an object is created, it loads the model, movie data, TMDB links, cache, and embedding vectors.

## 3. Constructor: `__init__`

```python
base_dir = Path(__file__).resolve().parents[1]
```

This finds the project root directory. Since this file is inside `src`, `parents[1]` points to the repo root.

Then it builds paths:

```python
self.model_path = base_dir / "model" / "ncf_best.keras"
self.movies_path = base_dir / "data" / "movies_encoded.csv"
self.links_path = base_dir / "data" / "ml-20m" / "links.csv"
self.poster_cache_path = base_dir / "data" / "poster_cache.json"
```

These point to:

- trained model
- encoded movie metadata
- MovieLens-to-TMDB ID mapping
- local cache for poster/details data

TMDB settings are also loaded:

```python
self.tmdb_image_base_url = "https://image.tmdb.org/t/p"
self.tmdb_read_access_token = os.getenv("TMDB_READ_ACCESS_TOKEN")
self.tmdb_api_key = os.getenv("TMDB_API_KEY")
```

The app can use either a TMDB bearer token or API key if one is available in environment variables.

## 4. Load Model and Movies

```python
self.model = tf.keras.models.load_model(self.model_path, compile=False)
```

This loads the trained Keras model from `model/ncf_best.keras`.

`compile=False` means the model is loaded for prediction/embedding use, not for more training.

```python
self.movies = pd.read_csv(self.movies_path)
self.movies = self.movies.dropna(subset=["movieId_enc"])
self.movies["movieId_enc"] = self.movies["movieId_enc"].astype(int)
```

This loads `movies_encoded.csv`, removes rows without encoded movie IDs, and ensures `movieId_enc` is an integer.

The encoded movie ID is important because it lines up with the movie embedding matrix inside the model.

## 5. Load TMDB Links and Poster Cache

```python
self._load_movie_links()
self.poster_cache = self._load_poster_cache()
```

`_load_movie_links()` adds TMDB IDs to the movie dataframe.

`_load_poster_cache()` reads cached TMDB poster/detail data from `data/poster_cache.json`.

The cache avoids repeatedly calling the TMDB API for the same movie.

## 6. Extract Movie Embeddings

```python
self.movie_embeddings = self.model.get_layer("movie_embedding").get_weights()[0]
```

The trained model has a layer named `movie_embedding`.

That layer stores a vector for every movie. Movies with similar learned patterns should have vectors that point in similar directions.

Example idea:

```text
The Matrix -> [0.12, -0.44, 0.89, ...]
Inception   -> [0.18, -0.39, 0.82, ...]
```

These vectors are what the recommender uses for similarity.

## 7. Normalize Embeddings

```python
norms = np.linalg.norm(self.movie_embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1
self.movie_embeddings_norm = self.movie_embeddings / norms
```

This normalizes each movie vector to length `1`.

After normalization, cosine similarity can be calculated with a simple dot product:

```python
similarities = self.movie_embeddings_norm @ selected_vector
```

## 8. `_load_movie_links`

```python
def _load_movie_links(self):
```

This method loads `links.csv`, which connects MovieLens movie IDs to TMDB IDs.

If the file does not exist:

```python
self.movies["tmdbId"] = pd.NA
return
```

The app still works, but posters/details will not be available.

If the file exists:

```python
links = pd.read_csv(self.links_path, usecols=["movieId", "tmdbId"])
links = links.dropna(subset=["tmdbId"])
links["tmdbId"] = links["tmdbId"].astype(int)
self.movies = self.movies.merge(links, on="movieId", how="left")
```

The movie metadata is merged with TMDB IDs.

## 9. `_load_poster_cache`

```python
def _load_poster_cache(self):
```

This method loads cached TMDB details from `poster_cache.json`.

If the file is missing, unreadable, invalid JSON, or not a dictionary, it returns an empty dictionary.

That keeps the recommender running even if the cache has a problem.

## 10. `_save_poster_cache`

```python
def _save_poster_cache(self):
```

This writes the current poster/details cache back to disk.

The method catches `OSError`, so a cache write failure does not crash recommendations.

## 11. `_parse_genres`

```python
def _parse_genres(self, genres):
```

MovieLens stores genres as one string:

```text
Action|Adventure|Sci-Fi
```

This method converts that into a Python set:

```python
{"Action", "Adventure", "Sci-Fi"}
```

If genres are missing or listed as `(no genres listed)`, it returns an empty set.

Sets make it easy to find shared genres:

```python
selected_genres.intersection(candidate_genres)
```

## 12. `_nullable_int`

```python
def _nullable_int(self, value):
```

This converts a value to `int`, unless it is missing.

Missing values become `None`, which is easier to serialize into API responses.

## 13. `_poster_url`

```python
def _poster_url(self, poster_path, size="w342"):
```

TMDB returns poster paths like:

```text
/abc123.jpg
```

This method turns that into a full image URL:

```text
https://image.tmdb.org/t/p/w342/abc123.jpg
```

If there is no poster path, it returns `None`.

## 14. `_fetch_tmdb_details`

```python
def _fetch_tmdb_details(self, tmdb_id):
```

This method gets poster and detail data for a movie from TMDB.

First it checks whether the movie has a TMDB ID. If not, it returns an empty dictionary.

Then it checks the local cache:

```python
if cache_key in self.poster_cache:
```

If enough detail is already cached, it returns cached data immediately.

If there is no API token or API key:

```python
if not self.tmdb_read_access_token and not self.tmdb_api_key:
    return cached_details
```

The app still works, but without new TMDB detail fetching.

If credentials exist, it calls:

```text
https://api.themoviedb.org/3/movie/{tmdb_id}
```

Then it extracts useful fields:

- poster path
- overview
- tagline
- release date
- runtime
- vote average
- vote count
- popularity
- language
- status

Finally, it saves the result in the cache.

## 15. `_serialize_movie`

```python
def _serialize_movie(self, row, poster_size="w342"):
```

This converts a movie row into a clean dictionary suitable for an API response.

It includes local movie fields:

- `movieId`
- `movieId_enc`
- `title`
- `genres`

It also includes TMDB fields:

- `tmdbId`
- `posterUrl`
- `overview`
- `tagline`
- `releaseDate`
- `runtime`
- `voteAverage`
- `voteCount`
- `popularity`
- `originalLanguage`
- `status`
- `tmdbUrl`

This is the method that turns dataframe rows into frontend-friendly JSON.

## 16. `get_movies`

```python
def get_movies(self, limit=25, page=1, query=None):
```

This returns a paginated list of movies.

If `query` is provided:

```python
filtered = self.movies[
    self.movies["title"].str.contains(query, case=False, na=False, regex=False)
]
```

It searches movie titles without caring about uppercase/lowercase.

Pagination is done with:

```python
start = (page - 1) * limit
end = start + limit
result = filtered.iloc[start:end]
```

Then each movie row is serialized into a dictionary.

The method returns:

```python
movies, total
```

`movies` is the current page. `total` is the total number of matching movies.

## 17. `get_movie_details`

```python
def get_movie_details(self, movie_id_enc: int):
```

This finds one movie by its encoded movie ID.

If no movie is found, it returns `None`.

If found, it serializes and returns the movie details.

## 18. `recommend_similar_movies`

```python
def recommend_similar_movies(self, movie_id_enc: int, top_n: int = 10):
```

This is the main recommendation method.

It takes one encoded movie ID and returns movies similar to it.

## 19. Validate Selected Movie

```python
selected_movie = self.movies[self.movies["movieId_enc"] == movie_id_enc]
```

If the movie does not exist, the method returns:

```python
{
    "success": False,
    "message": "...",
    "movie": None,
    "recommendations": []
}
```

This prevents crashes when the API receives a bad movie ID.

## 20. Get Selected Movie Genres

```python
selected_genres = self._parse_genres(selected_movie["genres"])
```

The selected movie's genres are converted into a set so genre overlap can be calculated.

## 21. Calculate Embedding Similarity

```python
selected_vector = self.movie_embeddings_norm[movie_id_enc]
similarities = self.movie_embeddings_norm @ selected_vector
```

The selected movie's embedding vector is compared with every other movie embedding.

Because the vectors were normalized earlier, this dot product acts like cosine similarity.

Higher score means the model thinks the movies are more similar based on learned rating behavior.

## 22. Build Candidate Movies

```python
candidates = self.movies.copy()
candidates["embedding_score"] = candidates["movieId_enc"].apply(
    lambda idx: float(similarities[int(idx)])
)
```

This creates a candidate dataframe and attaches an embedding similarity score to each movie.

Then it removes the selected movie itself:

```python
candidates = candidates[candidates["movieId_enc"] != movie_id_enc].copy()
```

Without this, the top recommendation would usually be the same movie the user clicked.

## 23. Calculate Genre Score

```python
for genre_text in candidates["genres"]:
    candidate_genres = self._parse_genres(genre_text)
    shared_genres = selected_genres.intersection(candidate_genres)
```

For every candidate movie, the code checks which genres it shares with the selected movie.

The genre score is:

```python
genre_score = len(shared_genres) / len(selected_genres)
```

Example:

```text
Selected movie genres: Action, Adventure, Sci-Fi
Candidate shared:      Action, Sci-Fi
genre_score = 2 / 3 = 0.6667
```

The shared genre names are also saved so they can be shown in the response.

## 24. Calculate Final Hybrid Score

```python
candidates["final_score"] = (
    0.75 * candidates["embedding_score"] +
    0.25 * candidates["genre_score"]
)
```

This combines two recommendation signals:

- `embedding_score`: learned from user rating behavior
- `genre_score`: based on genre overlap

The embedding score has more weight, so the trained model drives most of the recommendation.

The genre score helps keep recommendations more visibly related to the selected movie.

## 25. Pick Top Recommendations

```python
recommendations = candidates.sort_values(
    by="final_score",
    ascending=False
).head(top_n)
```

Candidates are sorted from best to worst by final score.

Only the top `top_n` movies are returned.

## 26. Build Response

```python
for row in recommendations.itertuples():
    movie = self._serialize_movie(row)
    movie.update({
        "shared_genres": row.shared_genres,
        "embedding_score": round(float(row.embedding_score), 4),
        "genre_score": round(float(row.genre_score), 4),
        "final_score": round(float(row.final_score), 4)
    })
    result.append(movie)
```

Each recommended movie is converted into a dictionary, then score information is added.

The selected movie is serialized separately with a larger poster size:

```python
movie_info = self._serialize_movie(selected_movie, poster_size="w500")
```

The final response looks like:

```python
{
    "success": True,
    "movie": movie_info,
    "top_n": top_n,
    "recommendations": result
}
```

## Overall Flow

```text
load trained model
        |
        v
load movies_encoded.csv
        |
        v
load TMDB links and cache
        |
        v
extract movie_embedding layer
        |
        v
normalize movie vectors
        |
        v
for selected movie:
    compare embedding similarity
    compare genre overlap
    combine scores
    return top movies
```

This recommender does not directly predict a user's rating. Instead, it uses the trained movie embeddings to find movies similar to a selected movie, then improves the ranking with genre overlap.
