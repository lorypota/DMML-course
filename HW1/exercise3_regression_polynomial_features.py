import numpy as np
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing()
print(california.DESCR)
D = california.data
y = california.target
n,d = D.shape
print(n,d)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
scaler = StandardScaler()
aff = PolynomialFeatures(2,include_bias=True)
D_normalized = scaler.fit_transform(D)
X = aff.fit_transform(D_normalized)
aff.get_feature_names_out(california.feature_names)

print(f"Number of samples: {n}")
print(f"Number of features: {d}")
print(f"Design matrix shape: {X.shape}")