import numpy as np
import sklearn
from sklearn.datasets import fetch_20newsgroups
from exercise5_naive_bayes import *

categories = ['rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

#5a
probabilities = proportional_probability(categories=categories, target=train.target)
for category in categories:
    print(f'Probability of {category}: {probabilities[category]}')

#5b
