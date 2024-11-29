from sklearn import datasets, metrics, svm
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split



#7e
def get_best_combination_and_score(x,y,gamma_values,c_values,k_flod_cross_validation):
    param_grid = {
    'C': c_values, 
    'gamma': gamma_values
    }
    svc = SVC(kernel='rbf')
    #look for the parameters
    grid_search = GridSearchCV(
    estimator=svc,
    param_grid=param_grid,
    cv=k_flod_cross_validation,
    scoring='accuracy',
    verbose=1
    )
    grid_search.fit(x, y)#train model
    return grid_search.best_params_,grid_search.best_score_