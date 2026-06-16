import pickle
import os
import numpy as np
import pandas as pd
import tensorflow as tf

MODEL_PATH = "model/ncf_best.keras"
ENCODER_PATH="data/encoders.pkl"
MOVIES_PATH="data/movies_encoded.csv"
TRAIN_PATH="data/train.csv"
TEST_PATH="data/test.csv"


def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH,compile=False)
    with open(ENCODER_PATH,"rb") as f:
        encoders = pickle.load(f)
    
    movies = pd.read_csv(MOVIES_PATH)
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    ratings = pd.concat([train,test],ignore_index=True)

    return model,encoders,movies,ratings

def recommend_movies(user_id,top_n=10):
    model,encoders,movies, ratings = load_assets()

    user2idx = encoders["user2idx"]

    if user_id not in user2idx:
        print(f"User ID {user_id} not found in the dataset")
        return

    user_idx = user2idx[user_id]

    # find movies rated by user
    rated_movies = set(
     ratings.loc[ratings["userId"] == user_idx, "movie"].values
     )

     #not rated movies
    candidate_movies = movies[~movies["movieId_enc"].isin(rated_movies)].copy()

    if candidate_movies.empty:
        print("all movies are rated")
        return
    movie_indices = candidate_movies["movieId_enc"].values.astype(np.int32)

    user_array = np.full(
        shape=(len(movie_indices),1),
        fill_value = user_idx,
        dtype=np.int32
    )

    movie_array = movie_indices.reshape(-1,1)

    predictions = model.predict(
        {
            "user_input":user_array,
            "movie_input":movie_array
        },
        batch_size=4096,
        verbose=0
    ).flatten()

    candidate_movies["predicted_rating"] = predictions*5.0

    recommendations = candidate_movies.sort_values(by="predicted_rating",ascending=False).head(top_n)

    print(f"\nTop {top_n} recommendations for userId {user_id}:\n")
    for i,row in enumerate(recommendations.itertuples(),start=1):
        print(
            f"{i}. {row.title} "
            f"| {row.genres} "
            f"| predicted rating: {row.predicted_rating:.2f}"
        )

if __name__ == "__main__":
    user_id = int(input("Enter userId: "))
    recommend_movies(user_id,top_n=10)
