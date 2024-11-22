import numpy as np

def get_average_bias_squared(models,true_f,x_0):
    expectancy_model_x_0=sum(model(x_0) for model in models)/len(models)
    return (true_f(x_0)-expectancy_model_x_0)**2
    
def get_average_variance(models,true_f,x_0):
    expectancy_model_x_0=sum(model(x_0) for model in models)/len(models)
    return sum((expectancy_model_x_0-model(x_0))**2 for model in models)/len(models)