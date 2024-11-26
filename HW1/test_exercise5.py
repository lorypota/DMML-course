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
probabilities = proportional_probabilities(categories=categories, target=train.target)
for i, category in enumerate(categories):
    print(f'Probability of {category}, y={i}: {probabilities[i]}')
print('\n')

#5b
for i, category in enumerate(categories):
    print(f"log(p('naive' | y={i})) = {log_laplace_proabability_word_given_target(
        "naive", i, D_train, train, vectorizer, 10**-5)}")
print('\n')

#5c
def compute_posterior(word, category):
    return compute_posterior_probability(
        word=word,
        category=category,
        matrix=D_train,
        dataset=train,
        vectorizer=vectorizer,
        categories=categories
    )

print(f"p(y = 0 | x_auto = 1) = {compute_posterior('auto', 0)}")
print(f"p(y = 1 | x_motorcycles = 1) = {compute_posterior('motorcycles', 1)}")
print(f"p(y = 2 | x_baseball = 1) = {compute_posterior('baseball', 2)}")
print(f"p(y = 3 | x_hockey = 1) = {compute_posterior('hockey', 3)}")
