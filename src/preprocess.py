import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os

# Task1: get data from data folder
ratings = pd.read_csv("data/ml-20m/ratings.csv")
movies = pd.read_csv("data/ml-20m/movies.csv")

print(ratings)
print(movies)