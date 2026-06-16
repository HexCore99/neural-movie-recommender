# preprocess.ipynb Explained

This notebook prepares the raw MovieLens data so the neural network can train on it. The main outputs are:

- `../data/train.csv`
- `../data/test.csv`
- `../data/encoders.pkl`
- `../data/movies_encoded.csv`

## 1. Imports

```python
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
```

`pandas` is used for reading and transforming CSV files. `pickle` saves Python dictionaries such as the user/movie encoders. `os` creates output folders. `train_test_split` splits the ratings into training and testing sets.

## 2. Load Raw Data

```python
ratings = pd.read_csv("../data/ml-20m/ratings.csv")
movies = pd.read_csv("../data/ml-20m/movies.csv")
ratings = ratings.drop(columns=["timestamp"])
```

The notebook loads the MovieLens ratings and movie metadata.

`ratings.csv` contains user ratings, usually with columns like:

- `userId`
- `movieId`
- `rating`
- `timestamp`

The `timestamp` column is removed because this model only learns from user, movie, and rating values.

`movies.csv` contains movie information, usually:

- `movieId`
- `title`
- `genres`

The `display(...)` calls show sample rows so you can inspect whether the files loaded correctly.

## 3. Inspect Dataset

```python
print("Ratings shape:", ratings.shape)
print("MOvies shape:", movies.shape)
print("unique users:", ratings["userId"].unique())
print("Unique rated movies:", ratings["movieId"].unique())
print("\nRating min:", ratings.min())
print("\nRating max:", ratings.max())
print("\nMissing Values in Ratings:", ratings.isnull().sum())
print("\nMissing Values in Movies", movies.isnull().sum())
```

This section checks the size and quality of the dataset.

It prints:

- number of rating rows and movie rows
- unique users
- unique rated movies
- minimum values
- maximum values
- missing values in both dataframes

This is a sanity check before generating training files.

## 4. Encode Users and Movies

```python
user_ids = ratings["userId"].unique()
movie_ids = ratings["movieId"].unique()
```

The raw MovieLens IDs are not ideal for embedding layers. They can be large and non-contiguous, for example `movieId = 79132`.

The model needs compact integer indexes:

```text
0, 1, 2, 3, ...
```

So the notebook creates two dictionaries:

```python
user2idx = {}
for idx, uid in enumerate(user_ids):
    user2idx[uid] = idx

movie2idx = {}
for idx, mid in enumerate(movie_ids):
    movie2idx[mid] = idx
```

Example:

```text
userId 10 -> user 0
userId 25 -> user 1

movieId 1   -> movie 0
movieId 318 -> movie 1
```

Then the encoded IDs are added to the ratings dataframe:

```python
ratings["user"] = ratings["userId"].map(user2idx)
ratings["movie"] = ratings["movieId"].map(movie2idx)
```

The model trains on `user` and `movie`, not the original `userId` and `movieId`.

## 5. Normalize Ratings

```python
ratings["rating_norm"] = ratings["rating"] / 5.0
```

MovieLens ratings are on a 0.5 to 5.0 scale. This line converts them to a 0.0 to 1.0 scale.

Examples:

```text
5.0 -> 1.0
4.0 -> 0.8
2.5 -> 0.5
0.5 -> 0.1
```

This matters because the model in `train.py` uses a final `sigmoid` layer, which outputs values between `0` and `1`.

## 6. Count Users, Movies, and Ratings

```python
num_users = len(user2idx)
num_movies = len(movie2idx)
```

These values tell the model how many user embeddings and movie embeddings it needs.

For example, if there are `138493` users, the user embedding layer needs `138493` rows.

## 7. Train/Test Split

```python
train, test = train_test_split(
    ratings,
    test_size=0.2,
    random_state=32
)
```

This splits the ratings into:

- 80% training data
- 20% testing/validation data

`random_state=32` makes the split repeatable. If you run the notebook again with the same data, you should get the same split.

## 8. Save Train and Test CSV Files

```python
os.makedirs("../data", exist_ok=True)
train.to_csv("../data/train.csv", index=False)
test.to_csv("../data/test.csv", index=False)
```

The processed train and test datasets are saved to disk.

These files are later loaded by `train.py`:

```python
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
```

## 9. Reload and Preview Train/Test Files

```python
test = pd.read_csv("../data/test.csv")
train = pd.read_csv("../data/train.csv")

display(test.head(20))
display(train.head(20))
```

This confirms that the CSV files were written correctly and can be read back.

## 10. Save Encoders

```python
with open("../data/encoders.pkl", "wb") as f:
    pickle.dump({
        "user2idx": user2idx,
        "movie2idx": movie2idx,
        "num_users": num_users,
        "num_movies": num_movies
    }, f)
```

This saves the mapping dictionaries and counts.

`train.py` uses `num_users` and `num_movies` to build embedding layers.

Recommendation code uses `movie2idx` or encoded movie IDs to connect real movie data to model inputs.

## 11. Create movies_encoded.csv

```python
movies["movieId_enc"] = movies["movieId"].map(movie2idx)
movies = movies.dropna(subset=["movieId_enc"])
movies["movieId_enc"] = movies["movieId_enc"].astype(int)
movies.to_csv("../data/movies_encoded.csv", index=False)
```

This adds the model's encoded movie ID to the movie metadata table.

`movies.csv` has human-readable data like title and genres. The model uses encoded movie indexes. This file connects both worlds.

Example output:

```text
movieId,title,genres,movieId_enc
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy,0
```

`dropna(...)` removes movies that were never rated and therefore do not exist in `movie2idx`.

`astype(int)` converts the encoded movie ID from float to integer after mapping.

## 12. Completion Message

```python
print("Preprocessing Complete.")
print("Saved train.csv, test.csv, encoders.pkl, movies_encoded.csv")
```

This prints a final confirmation that preprocessing finished and the required files were saved.

## Overall Flow

```text
raw ratings.csv + movies.csv
        |
        v
drop timestamp
        |
        v
encode userId/movieId into compact integer IDs
        |
        v
normalize rating to rating_norm
        |
        v
split into train.csv and test.csv
        |
        v
save encoders.pkl
        |
        v
save movies_encoded.csv
```

The output of this notebook becomes the input for model training and recommendation.
