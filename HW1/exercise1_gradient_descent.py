import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Objective function
def f(u, v, b):
    term1 = -np.log(sigmoid(u + b))
    term2 = -np.log(sigmoid(v + b))
    term3 = -np.log(sigmoid(-u / 2 - v / 2 - b))
    term4 = (u**2 + v**2 + b**2) / 100
    return term1 + term2 + term3 + term4

# Gradient of f(u, v, b)
def grad_f(u, v, b):
    df_du = -sigmoid(-u-b) + sigmoid(-u/2 -v/2 -b)*np.exp(u/2 +v/2 +b)/2 + u/50
    df_dv = -sigmoid(-v-b) + sigmoid(-u/2 -v/2 -b)*np.exp(u/2 +v/2 +b)/2 + v/50
    df_db = -sigmoid(-u-b) -sigmoid(-v-b) + sigmoid(-u/2 -v/2 -b)*np.exp(u/2 +v/2 +b) + b/50
    
    return np.array([df_du, df_dv, df_db])

# Gradient Descent Algorithm
def gradient_descent(f, grad_f, eta, initial_point, max_iter=100):
    u, v, b = initial_point
    points = [(u, v, b)]
    values = [f(u, v, b)]

    for t in range(max_iter):
        grad = grad_f(u, v, b)
        step_size = eta(t)
        u, v, b = np.array([u, v, b]) - step_size * grad
        points.append((u, v, b))
        values.append(f(u, v, b))
    
    return points, values

# Step-size strategies
def eta_const(t, c=0.2):
    return c

def eta_sqrt(t, c=0.5):
    return c / np.sqrt(t + 1)

def eta_multistep(t, milestones=[30, 50, 80], c=0.5, eta_init=1.0):
    step = eta_init
    for milestone in milestones:
        if t >= milestone:
            step *= c
    return step

