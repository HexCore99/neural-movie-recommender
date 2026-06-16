# train.py Explained

This script trains a Neural Collaborative Filtering style model. It learns user embeddings and movie embeddings from rating data, then saves the best and final model files.

The main inputs are:

- `data/train.csv`
- `data/test.csv`
- `data/encoders.pkl`

The main outputs are:

- `model/ncf_best.keras`
- `model/ncf_final.keras`
- `model/training_history.csv`

## 1. Imports

```python
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import os
```

These libraries are used for:

- reading CSV files with `pandas`
- numerical operations with `numpy`
- loading saved encoders with `pickle`
- building/training the neural network with TensorFlow/Keras
- creating output folders with `os`

## 2. Define Constants

```python
EMBEDDING_DIM = 32
BATCH_SIZE = 4096
EPOCHS = 10
LEARNING_RATE = 0.001
```

These values control training.

`EMBEDDING_DIM = 32` means every user and every movie will be represented by a vector of 32 numbers.

`BATCH_SIZE = 4096` means the model processes 4096 rating rows at a time.

`EPOCHS = 10` means the model can pass through the training data up to 10 times.

`LEARNING_RATE = 0.001` controls how large each optimizer update is.

## 3. Define File Paths

```python
TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
ENCODER_PATH = "data/encoders.pkl"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
```

The script expects preprocessed files from `preprocess.ipynb`.

`os.makedirs(...)` ensures the `model` folder exists before saving model files.

## 4. GPU Check

```python
gpus = tf.config.list_physical_devices("GPU")
print(f"Gpus available: {gpus}")
```

This checks whether TensorFlow can see a GPU.

If a GPU exists:

```python
tf.config.experimental.set_memory_growth(gpus[0], True)
```

Memory growth tells TensorFlow to allocate GPU memory gradually instead of grabbing most of it immediately.

If TensorFlow cannot set memory growth, the exception is printed instead of crashing the script.

## 5. Load Train/Test Data

```python
test = pd.read_csv(TEST_PATH, usecols=["user", "movie", "rating_norm"], dtype={...})
train = pd.read_csv(TRAIN_PATH, usecols=["user", "movie", "rating_norm"], dtype={...})
```

The script loads only the columns needed for training:

- `user`: encoded user ID
- `movie`: encoded movie ID
- `rating_norm`: normalized rating target

It also sets efficient data types:

- `user` as `np.int32`
- `movie` as `np.int32`
- `rating_norm` as `np.float32`

This reduces memory usage, which matters because the MovieLens 20M dataset is large.

## 6. Load Encoders

```python
with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)

num_users = encoders["num_users"]
num_movies = encoders["num_movies"]
```

The script loads `encoders.pkl`, created during preprocessing.

`num_users` and `num_movies` are needed to define the size of the embedding layers.

For example:

```python
tf.keras.layers.Embedding(input_dim=num_users, output_dim=embedding_dim)
```

means Keras creates one trainable vector per user.

## 7. Print Dataset Info

```python
print(f"Users: {num_users}")
print(f"Movies: {num_movies}")
print(f"Train Rows: {len(train)}")
print(f"Test Rows: {len(test)}")
```

This confirms the model is using the expected number of users, movies, and rating rows.

## 8. Prepare Training Inputs

```python
X_train = {
    "user_input": train["user"].values.reshape(-1, 1),
    "movie_input": train["movie"].values.reshape(-1, 1)
}
Y_train = train["rating_norm"].values
```

The model has two input layers:

- `user_input`
- `movie_input`

So `X_train` is a dictionary with matching names.

`reshape(-1, 1)` turns a flat array into a column shape Keras expects.

`Y_train` is the value the model learns to predict: the normalized rating.

Example:

```text
user_input = 10
movie_input = 452
target rating_norm = 0.8
```

This means: user 10 rated movie 452 as 4 stars, because `0.8 * 5 = 4.0`.

## 9. Prepare Test Inputs

```python
X_test = {
    "user_input": test["user"].values.reshape(-1, 1),
    "movie_input": test["movie"].values.reshape(-1, 1)
}
Y_test = test["rating_norm"].values
```

This prepares the validation/test data in the same format as the training data.

The model trains on `X_train, Y_train` and validates on `X_test, Y_test`.

## 10. Build NCF Model

```python
def build_ncf_model(num_users, num_movies, embedding_dim):
```

This function builds the neural network.

NCF means Neural Collaborative Filtering. The idea is:

```text
user ID + movie ID -> embeddings -> dense layers -> predicted rating
```

## 11. Input Layers

```python
user_input = tf.keras.Input(shape=(1,), name="user_input")
movie_input = tf.keras.Input(shape=(1,), name="movie_input")
```

The model takes two numbers for each row:

- encoded user ID
- encoded movie ID

The names must match the keys in `X_train` and `X_test`.

## 12. Embedding Layers

```python
user_embedding = tf.keras.layers.Embedding(
    input_dim=num_users,
    output_dim=embedding_dim,
    name="user_embedding"
)(user_input)
```

This creates a trainable vector for each user.

```python
movie_embedding = tf.keras.layers.Embedding(
    input_dim=num_movies,
    output_dim=embedding_dim,
    name="movie_embedding"
)(movie_input)
```

This creates a trainable vector for each movie.

During training, the model learns which users and movies are similar based on ratings.

## 13. Flatten Embeddings

```python
user_vector = tf.keras.layers.Flatten()(user_embedding)
movie_vector = tf.keras.layers.Flatten()(movie_embedding)
```

Embedding layers output a 3D tensor. `Flatten()` converts each embedding into a normal vector so it can be passed into dense layers.

## 14. Combine User and Movie Vectors

```python
x = tf.keras.layers.Concatenate()([user_vector, movie_vector])
```

This joins the user vector and movie vector together.

If each vector has 32 numbers, the combined vector has 64 numbers.

## 15. Dense Layers

```python
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
```

These layers learn patterns between users, movies, and ratings.

`relu` helps the network learn non-linear relationships.

`Dropout` randomly disables some neurons during training. This helps reduce overfitting.

## 16. Output Layer

```python
output = tf.keras.layers.Dense(1, activation="sigmoid", name="rating_output")(x)
```

The model outputs one number: the predicted normalized rating.

`sigmoid` forces the prediction to be between `0` and `1`.

That matches `rating_norm`, which was created by dividing the original rating by `5`.

To convert back to stars:

```python
predicted_stars = prediction * 5.0
```

## 17. Create the Model

```python
model = tf.keras.Model(
    inputs=[user_input, movie_input],
    outputs=output
)
```

This creates the final Keras model object.

The model takes two inputs and returns one predicted rating.

## 18. Build and Show Summary

```python
model = build_ncf_model(num_users, num_movies, EMBEDDING_DIM)
model.summary()
```

This builds the model using the actual number of users and movies, then prints the layer structure and parameter count.

## 19. Compile the Model

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse")
    ]
)
```

Compiling defines how the model learns.

`Adam` is the optimizer. It updates model weights during training.

`mse` means Mean Squared Error. It penalizes larger prediction errors more strongly.

Metrics:

- `mae`: average absolute error
- `rmse`: root mean squared error

These are calculated on normalized ratings during Keras training.

## 20. Define Callbacks

Callbacks are helper tools that run during training.

### ModelCheckpoint

```python
tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "ncf_best.keras"),
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)
```

This saves the best model based on validation loss.

If epoch 4 is better than epoch 3, it saves epoch 4. If epoch 5 is worse, it does not overwrite the best model.

### EarlyStopping

```python
tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)
```

This stops training if validation loss does not improve for 3 epochs.

`restore_best_weights=True` means the model returns to the best epoch's weights before evaluation/saving.

### ReduceLROnPlateau

```python
tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)
```

If validation loss stops improving, this lowers the learning rate.

That can help the model make smaller, more careful updates near the end of training.

## 21. Train the Model

```python
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)
```

This starts training.

The model learns from:

```python
X_train -> Y_train
```

And checks performance on:

```python
X_test -> Y_test
```

The training results for each epoch are stored in `history`.

## 22. Predict on Test Data

```python
y_pred = model.predict(X_test, batch_size=BATCH_SIZE).flatten()
```

This generates predicted normalized ratings for the test set.

`.flatten()` turns the output into a simple 1D array.

## 23. Convert Back to 5-Star Scale

```python
y_pred_scaled = y_pred * 5.0
y_test_scaled = Y_test * 5.0
```

The model predicts on the normalized 0 to 1 scale.

These lines convert predictions and real ratings back to the original 0.5 to 5.0 style scale.

This makes RMSE and MAE easier to understand.

## 24. Calculate Final Metrics

```python
rmse = np.sqrt(np.mean((y_pred_scaled - y_test_scaled) ** 2))
mae = np.mean(np.abs(y_pred_scaled - y_test_scaled))
```

`RMSE` measures typical prediction error, with bigger errors punished more heavily.

`MAE` measures average absolute error.

Example:

```text
Predicted: 4.2
Actual:    5.0
Error:     0.8
```

The printed targets are:

```text
RMSE target: < 1.0
MAE target:  < 0.8
```

## 25. Save Final Model

```python
final_model_path = os.path.join(MODEL_DIR, "ncf_final.keras")
model.save(final_model_path)
```

This saves the final model after training finishes.

There are two model files:

- `ncf_best.keras`: best validation model saved by callback
- `ncf_final.keras`: final model state after training

Because early stopping restores best weights, the final model may also contain the best weights.

## 26. Save Training History

```python
history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(MODEL_DIR, "training_history.csv"), index=False)
```

This saves epoch-by-epoch training metrics.

The CSV can include columns like:

- `loss`
- `mae`
- `rmse`
- `val_loss`
- `val_mae`
- `val_rmse`
- `learning_rate`

This is useful for plotting training curves later.

## Overall Flow

```text
load train.csv and test.csv
        |
        v
load encoders.pkl
        |
        v
prepare user/movie inputs and rating targets
        |
        v
build user and movie embedding model
        |
        v
train with validation callbacks
        |
        v
evaluate on 5-star scale
        |
        v
save best model, final model, and training history
```

The trained model can then be used by recommendation code to either predict ratings or extract learned movie embeddings.
