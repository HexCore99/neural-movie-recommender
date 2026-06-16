from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import requests
import tensorflow as tf


class MovieRecommender:
    def __init__(self):
        base_dir = Path(__file__).resolve().parents[1]

        self.model_path = base_dir / "model" / "ncf_best.keras"
        self.movies_path = base_dir / "data" / "movies_encoded.csv"
        self.links_path = base_dir / "data" / "ml-20m" / "links.csv"
        self.poster_cache_path = base_dir / "data" / "poster_cache.json"
        self.tmdb_image_base_url = "https://image.tmdb.org/t/p"
        self.tmdb_read_access_token = os.getenv("TMDB_READ_ACCESS_TOKEN")
        self.tmdb_api_key = os.getenv("TMDB_API_KEY")

        print("Loading movie recommendation model...")

        self.model = tf.keras.models.load_model(self.model_path, compile=False)

        self.movies = pd.read_csv(self.movies_path)
        self.movies = self.movies.dropna(subset=["movieId_enc"])
        self.movies["movieId_enc"] = self.movies["movieId_enc"].astype(int)
        self._load_movie_links()
        self.poster_cache = self._load_poster_cache()

        # Get learned movie vectors from trained NCF model
        self.movie_embeddings = self.model.get_layer("movie_embedding").get_weights()[0]

        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(self.movie_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1

        self.movie_embeddings_norm = self.movie_embeddings / norms

        print("Movie recommender ready.")
        print("Movie embedding shape:", self.movie_embeddings.shape)

    def _load_movie_links(self):
        if not self.links_path.exists():
            self.movies["tmdbId"] = pd.NA
            return

        links = pd.read_csv(self.links_path, usecols=["movieId", "tmdbId"])
        links = links.dropna(subset=["tmdbId"])
        links["tmdbId"] = links["tmdbId"].astype(int)

        self.movies = self.movies.merge(links, on="movieId", how="left")

    def _load_poster_cache(self):
        if not self.poster_cache_path.exists():
            return {}

        try:
            with self.poster_cache_path.open("r", encoding="utf-8") as file:
                cache = json.load(file)
        except (OSError, json.JSONDecodeError):
            return {}

        return cache if isinstance(cache, dict) else {}

    def _save_poster_cache(self):
        try:
            self.poster_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.poster_cache_path.open("w", encoding="utf-8") as file:
                json.dump(self.poster_cache, file, indent=2, sort_keys=True)
        except OSError:
            pass

    def _parse_genres(self, genres):
        if pd.isna(genres) or genres == "(no genres listed)":
            return set()

        return set(genres.split("|"))

    def _nullable_int(self, value):
        if pd.isna(value):
            return None

        return int(value)

    def _poster_url(self, poster_path, size="w342"):
        if not poster_path:
            return None

        return f"{self.tmdb_image_base_url}/{size}{poster_path}"

    def _fetch_tmdb_details(self, tmdb_id):
        if tmdb_id is None:
            return {}

        cache_key = str(tmdb_id)
        cached_details = {}

        if cache_key in self.poster_cache:
            cached = self.poster_cache[cache_key]

            if isinstance(cached, dict):
                cached_details = cached

                if cached.get("overview") or cached.get("release_date"):
                    return cached
            else:
                cached_details = {"poster_path": cached}

        if not self.tmdb_read_access_token and not self.tmdb_api_key:
            return cached_details

        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        headers = {}
        params = {"language": "en-US"}

        if self.tmdb_read_access_token:
            headers["Authorization"] = f"Bearer {self.tmdb_read_access_token}"
        else:
            params["api_key"] = self.tmdb_api_key

        try:
            response = requests.get(url, headers=headers, params=params, timeout=8)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            return cached_details

        details = {
            "poster_path": data.get("poster_path"),
            "overview": data.get("overview"),
            "tagline": data.get("tagline"),
            "release_date": data.get("release_date"),
            "runtime": data.get("runtime"),
            "vote_average": data.get("vote_average"),
            "vote_count": data.get("vote_count"),
            "popularity": data.get("popularity"),
            "original_language": data.get("original_language"),
            "status": data.get("status"),
        }

        self.poster_cache[cache_key] = details
        self._save_poster_cache()

        return details

    def _serialize_movie(self, row, poster_size="w342"):
        if hasattr(row, "_asdict"):
            row = row._asdict()

        tmdb_id = self._nullable_int(row.get("tmdbId"))
        tmdb_details = self._fetch_tmdb_details(tmdb_id)
        poster_path = tmdb_details.get("poster_path")

        return {
            "movieId": int(row["movieId"]),
            "movieId_enc": int(row["movieId_enc"]),
            "title": row["title"],
            "genres": row["genres"],
            "tmdbId": tmdb_id,
            "posterUrl": self._poster_url(poster_path, poster_size),
            "overview": tmdb_details.get("overview"),
            "tagline": tmdb_details.get("tagline"),
            "releaseDate": tmdb_details.get("release_date"),
            "runtime": tmdb_details.get("runtime"),
            "voteAverage": tmdb_details.get("vote_average"),
            "voteCount": tmdb_details.get("vote_count"),
            "popularity": tmdb_details.get("popularity"),
            "originalLanguage": tmdb_details.get("original_language"),
            "status": tmdb_details.get("status"),
            "tmdbUrl": f"https://www.themoviedb.org/movie/{tmdb_id}" if tmdb_id else None
        }

    def get_movies(self, limit=25, page=1, query=None):
        if query:
            filtered = self.movies[
                self.movies["title"].str.contains(
                    query,
                    case=False,
                    na=False,
                    regex=False
                )
            ]
        else:
            filtered = self.movies

        total = len(filtered)
        start = (page - 1) * limit
        end = start + limit
        result = filtered.iloc[start:end]

        movies = []

        for row in result.itertuples():
            movies.append(self._serialize_movie(row))

        return movies, total

    def get_movie_details(self, movie_id_enc: int):
        selected = self.movies[self.movies["movieId_enc"] == movie_id_enc]

        if selected.empty:
            return None

        row = selected.iloc[0]

        return self._serialize_movie(row)

    def recommend_similar_movies(self, movie_id_enc: int, top_n: int = 10):
        selected_movie = self.movies[self.movies["movieId_enc"] == movie_id_enc]

        if selected_movie.empty:
            return {
                "success": False,
                "message": f"Movie with encoded ID {movie_id_enc} not found.",
                "movie": None,
                "recommendations": []
            }

        selected_movie = selected_movie.iloc[0]
        selected_genres = self._parse_genres(selected_movie["genres"])

        # Selected movie vector
        selected_vector = self.movie_embeddings_norm[movie_id_enc]

        # Cosine similarity with all movie vectors
        similarities = self.movie_embeddings_norm @ selected_vector

        candidates = self.movies.copy()

        candidates["embedding_score"] = candidates["movieId_enc"].apply(
            lambda idx: float(similarities[int(idx)])
        )

        # Remove clicked movie itself
        candidates = candidates[candidates["movieId_enc"] != movie_id_enc].copy()

        genre_scores = []
        shared_genres_list = []

        for genre_text in candidates["genres"]:
            candidate_genres = self._parse_genres(genre_text)
            shared_genres = selected_genres.intersection(candidate_genres)

            if len(selected_genres) == 0:
                genre_score = 0
            else:
                genre_score = len(shared_genres) / len(selected_genres)

            genre_scores.append(genre_score)
            shared_genres_list.append(sorted(list(shared_genres)))

        candidates["genre_score"] = genre_scores
        candidates["shared_genres"] = shared_genres_list

        # Hybrid score: learned embedding similarity + genre match
        candidates["final_score"] = (
            0.75 * candidates["embedding_score"] +
            0.25 * candidates["genre_score"]
        )

        recommendations = candidates.sort_values(
            by="final_score",
            ascending=False
        ).head(top_n)

        result = []

        for row in recommendations.itertuples():
            movie = self._serialize_movie(row)
            movie.update({
                "shared_genres": row.shared_genres,
                "embedding_score": round(float(row.embedding_score), 4),
                "genre_score": round(float(row.genre_score), 4),
                "final_score": round(float(row.final_score), 4)
            })
            result.append(movie)

        movie_info = self._serialize_movie(selected_movie, poster_size="w500")

        return {
            "success": True,
            "movie": movie_info,
            "top_n": top_n,
            "recommendations": result
        }
