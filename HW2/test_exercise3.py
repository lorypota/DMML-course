import pandas as pd
import numpy as np
import os
from exercise3_netflix import matrix_completion, average_squared_error, est_ratings

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

# Final preprocessed matrix
D = df_D.to_numpy()
n, d = D.shape
print("(n, d): (", n, ", ", d, ")")


X, Y = matrix_completion(D=D, n=n, d=d)
avg_sq_err = average_squared_error(D=D, X=X, Y=Y)
print(f"AVG squared error:{avg_sq_err}")
est_ratings(D, X, Y)