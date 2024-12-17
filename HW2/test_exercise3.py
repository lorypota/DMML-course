import pandas as pd
import numpy as np

# Load the data
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv', sep=',')

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

