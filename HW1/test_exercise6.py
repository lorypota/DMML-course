from sklearn.datasets import load_iris
from exercise6_decision_tree import *

iris = load_iris()
D, y = iris.data, iris.target

#6a
root_gini = gini_impurity(y)
print(f"Gini impurity at the root node: {root_gini}")

#6b
split_value = np.mean(D[:, 0])
L0_y, L1_y = y[D[:, 0] <= split_value], y[D[:, 0] > split_value]
cost = split_cost(L0_y, L1_y, y)
print(f"Cost of the split: {cost}")