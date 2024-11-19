import numpy as np
def f(x):
    return np.exp(x[0]-x[1]+1)+np.exp(x[1]-x[2]+2)+np.exp(x[2]-x[0]+3)
#2a
def argmin_x1(x):
    return (x[1]+x[2])/2 +1
def argmin_x2(x):
    return (x[0]+x[2])/2-0.5
def argmin_x3(x):
    return (x[0]+x[1])/2-0.5

#2b
def coordinate_descent(f,argmin,x_0,max_iter=100):
    num_coordinates=len(x_0)
    x_t=x_0
    for _ in range(max_iter):
        x_t_plus_1=[argmin[i](x_t) for i in range(num_coordinates)]
        x_t=x_t_plus_1
    return x_t