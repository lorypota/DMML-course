import numpy as np
import sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
#get the data
print('The scikit-learn version is {}.'.format(sklearn.__version__))
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
# print(california.DESCR)
D = california.data
y = california.target
n,d = D.shape

#Scale de date
scaler = StandardScaler()
D_normalized = scaler.fit_transform(D)


aff = PolynomialFeatures(2,include_bias=True)
X = aff.fit_transform(D_normalized)
aff.get_feature_names_out(california.feature_names)
#2.a
print(f"Number of samples: {n}")
print(f"Number of features: {d}")
print(f"Design matrix shape: {X.shape}")