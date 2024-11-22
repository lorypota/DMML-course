import numpy as np
from exercise4_bias_var_trade_off import *


def f_true(x):
    return np.tan(np.pi * x)

models=[lambda x:x+0.2, lambda x:3*x+0.3,lambda x:5*x+0.1]

print("expected bias squared:"+str(get_average_bias_squared(models,f_true,0)))

print("expected varience:"+str(get_average_variance(models,f_true,0)))
