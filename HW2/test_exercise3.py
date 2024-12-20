import pandas as pd
import numpy as np
import os
from exercise3_netflix import *

MOVIES_PATH = os.path.join("HW2", "ml-latest-small", "movies.csv")
RATINGS_PATH = os.path.join("HW2", "ml-latest-small", "ratings.csv")

# Load the data
movies = pd.read_csv(MOVIES_PATH)
ratings = pd.read_csv(RATINGS_PATH, sep=',')

# Convert sparse data representation into a matrix
# Fill unobserved entries with μ = 0
df_movie_ratings = ratings.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)

# Keep only movies with more than 100 ratings
keep_movie = np.sum(df_movie_ratings != 0, 0) > 100
df_D = df_movie_ratings.loc[:, keep_movie]

# Keep only users who have rated at least 5 movies
keep_user = np.sum(df_D != 0, 1) >= 5
df_D = df_D.loc[keep_user, :]

# Track filtered movieIDs and userIDs
filtered_movie_ids = df_D.columns
filtered_user_ids = df_D.index
movieId_to_title = dict(zip(movies['movieId'], movies['title']))

# Final preprocessed matrix
D = df_D.to_numpy()
n, d = D.shape
print("(n, d): (", n, ", ", d, ")")

# Check if saved matrices exist; otherwise compute and save them
X_file = "X_matrix.npy"
Y_file = "Y_matrix.npy"

if os.path.exists(X_file) and os.path.exists(Y_file):
    print("Loading X and Y matrices from files...")
    X = np.load(X_file)
    Y = np.load(Y_file)
else:
    print("Calculating X and Y matrices...")
    X, Y = matrix_completion(D=D, n=n, d=d)
    np.save(X_file, X)
    np.save(Y_file, Y)

# 3a
avg_sq_err = average_squared_error(D=D, X=X, Y=Y)
print(f"AVG squared error:{avg_sq_err}\n\n")

print(f"Estimation for first user and movie Finding Nemo (2003): {
    find_est_by_movie_id(D, X, Y, filtered_movie_ids, 6377)}")

print(f"Estimation for first user and movie Dark Knight, The (2008): {
    find_est_by_movie_id(D, X, Y, filtered_movie_ids, 58559)}")

print(f"Estimation for first user and movie Clueless (1995): {
    find_est_by_movie_id(D, X, Y, filtered_movie_ids, 39)}")

print(f"Estimation for first user and movie 2001: A Space Odyssey (1968): {
    find_est_by_movie_id(D, X, Y, filtered_movie_ids, 924)}")
