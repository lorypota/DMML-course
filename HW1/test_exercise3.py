import numpy as np
import sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from exercise3_regression_polynomial_features import *
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
#3.a
print(f"Number of samples: {n}")
print(f"Number of features: {d}")
print(f"Design matrix shape: {X.shape}")

#3.b
beta=get_beta_RSS_3_b(X,y)
beta_name_positions=aff.get_feature_names_out(california.feature_names)
indexes={}
for i,name in enumerate(beta_name_positions):
    indexes[name]=i
print("MedInc:"+str(beta[indexes["MedInc"]]))
print("MedInc AveBedrms:"+str(beta[indexes["MedInc AveBedrms"]]))
print("HouseAge AveBedrms:"+str(beta[indexes["HouseAge AveBedrms"]]))

#3.c
beta_c=get_beta_3_c(X,y,0.1)
print("----------------------------------------")
print("3c")
print("MedInc:"+str(beta_c[indexes["MedInc"]]))
print("MedInc AveBedrms:"+str(beta_c[indexes["MedInc AveBedrms"]]))
print("HouseAge AveBedrms:"+str(beta_c[indexes["HouseAge AveBedrms"]]))