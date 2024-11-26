import numpy as np
import sklearn
from sklearn.datasets import fetch_20newsgroups

categories = ['rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)
