import numpy as np
import sklearn
from sklearn.datasets import fetch_20newsgroups
from exercise5_naive_bayes import *
from sklearn.feature_extraction.text import CountVectorizer


categories = ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
vectorizer = CountVectorizer(
    stop_words="english", min_df=5, token_pattern="[^\W\d_]+", binary=True)
D_train = vectorizer.fit_transform(train.data)
D_test = vectorizer.transform(test.data)

#5a
probabilities = proportional_probability(
    categories=categories, target=train.target)
for i, category in enumerate(categories):
    print(f'Probability of {category}, y={i}: {probabilities[category]}')

#5b
for i, category in enumerate(categories):
    print(f"log(p('naive' | y={i})) = {log_laplace_proabability_word_given_target(
        "naive", i, D_train, train, vectorizer, 10**-5)}")
