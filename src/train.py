import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import os

#Task: Define Constants
EMBEDDING_DIM = 32
BATCH_SIZE = 4096
EPOCHS = 10
LEARNING_RATE = 0.001

TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"
ENCODER_PATH = "data/encoders.pkl"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

#Task: GPU check
gpus = tf.config.list_physical_devices("GPU")
print(f"Gpus available: {gpus}")

if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0],True)
        print("Gpu Memory Growth Enabled")
    except RuntimeError as e:
        print("Gpu memory growth could not be set:",e)


     #Task 7: Load created files and encoders

test = pd.read_csv(TEST_PATH,
                   usecols=["user","movie","rating_norm"],
                   dtype={
                       "user":np.int32,
                       "movie":np.int32,
                       "rating_norm":np.float32
                   })

train = pd.read_csv(TRAIN_PATH,
                   usecols=["user","movie","rating_norm"],
                   dtype={
                       "user":np.int32,
                       "movie":np.int32,
                       "rating_norm":np.float32
                   })
with open (ENCODER_PATH,"rb") as f:
    encoders = pickle.load(f)

num_users = encoders["num_users"]
num_movies = encoders["num_movies"]

print(f"Users: {num_users}")
print(f"Movies: {num_movies}")
print(f"Train Rows: {len(train)}")
print(f"Test Rows: {len(test)}")

#Task 8: Prepare Input
X_train = {
    "user_input":train["user"].values.reshape(-1,1),
    "movie_input":train["movie"].values.reshape(-1,1)
}
Y_train = train["rating_norm"].values


X_test= {
    "user_input":test["user"].values.reshape(-1,1),
    "movie_input":test["movie"].values.reshape(-1,1)
}
Y_test= test["rating_norm"].values



#Task 9:Build NCF Model
def build_ncf_model(num_users,num_movies,embedding_dim):
    user_input = tf.keras.Input(shape=(1,),name="user_input")
    movie_input = tf.keras.Input(shape=(1,),name="movie_input")

    user_embedding = tf.keras.layers.Embedding(
        input_dim=num_users,
        output_dim=embedding_dim,
        name="user_embedding"
    )(user_input)

    movie_embedding = tf.keras.layers.Embedding(
        input_dim=num_movies,
        output_dim=embedding_dim,
        name="movie_embedding"
    )(movie_input)

    user_vector = tf.keras.layers.Flatten()(user_embedding)
    movie_vector = tf.keras.layers.Flatten()(movie_embedding)

    x = tf.keras.layers.Concatenate()([user_vector,movie_vector])
    x = tf.keras.layers.Dense(128,activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64,activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32,activation="relu")(x)

    output = tf.keras.layers.Dense(1,activation="sigmoid",name="rating_output")(x)

    model = tf.keras.Model(
        inputs=[user_input,movie_input],
        outputs =output
    )
    return model

print("\nBuilding Model...")
model = build_ncf_model(num_users,num_movies,EMBEDDING_DIM)
model.summary()

#Task 10:Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="mse",
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse")
    ]
)


#Task 11:Define Callbacks

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "ncf_best.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


#Task 12: Train the Model. Let's Goooooooooooooooooo

print("\nVroom Vroom..........")
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test,Y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

#Task 13: prediction
print("\nEvaluating final model....")
y_pred = model.predict(X_test,batch_size=BATCH_SIZE).flatten()
y_pred_scaled = y_pred*5.0
y_test_scaled = Y_test*5.0

rmse = np.sqrt(np.mean((y_pred_scaled - y_test_scaled) ** 2))
mae = np.mean(np.abs(y_pred_scaled - y_test_scaled))

print("\n── Final Results ──")
print(f"RMSE: {rmse:.4f}  target: < 1.0")
print(f"MAE:  {mae:.4f}  target: < 0.8")


#Task 14: SAve Model and History

final_model_path=os.path.join(MODEL_DIR,"ncf_final.keras")
model.save(final_model_path)

history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(MODEL_DIR,"training_history.csv"),index=False)


print("\nTraining complete.")
print(f"Best model saved to:  {MODEL_DIR}/ncf_best.keras")
print(f"Final model saved to: {MODEL_DIR}/ncf_final.keras")
print(f"History saved to:     {MODEL_DIR}/training_history.csv")