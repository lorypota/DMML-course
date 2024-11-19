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
    sig1 = sigmoid(u + b)
    sig2 = sigmoid(v + b)
    sig3 = sigmoid(-u / 2 - v / 2 - b)
    
    # Partial derivatives
    df_du = -1/sig1*np.exp(-u-b)*sig1**2 - 1/sig3*np.exp(-u/2 -v/2 -b)*0.5*sig3**2 + (2*u)/100
    df_dv = -1/sig2*np.exp(-v-b)*sig2**2 - 1/sig3*np.exp(-u/2 -v/2 -b)*0.5*sig3**2 + (2*v)/100
    df_db = -1/sig1*sig1*np.exp(-u-b)*sig1**2 - 1/sig2*np.exp(-v-b)*sig2**2 - 1/sig3*sig3*np.exp(-u/2 -v/2 -b)*sig3**2 + (2*b)/100
    
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
    return c / (t + 1)

def eta_multistep(t, milestones=[30, 50, 80], c=0.5, eta_init=1.0):
    step = eta_init
    for milestone in milestones:
        if t >= milestone:
            step *= c
    return step

# Initialize parameters
initial_point = (4, 2, 1)
max_iter = 100

# Run gradient descent for each step-size strategy
results = {}

# Constant step size
points_const, values_const = gradient_descent(f, grad_f, eta=lambda t: eta_const(t, c=0.2), initial_point=initial_point, max_iter=max_iter)
results['const'] = (values_const[-1], min(values_const))

# Decreasing step size
points_sqrt, values_sqrt = gradient_descent(f, grad_f, eta=lambda t: eta_sqrt(t, c=0.5), initial_point=initial_point, max_iter=max_iter)
results['sqrt'] = (values_sqrt[-1], min(values_sqrt))

# Multi-step step size
points_multistep, values_multistep = gradient_descent(
    f, grad_f, eta=lambda t: eta_multistep(t, milestones=[30, 50, 80], c=0.5, eta_init=1.0), 
    initial_point=initial_point, max_iter=max_iter
)
results['multistep'] = (values_multistep[-1], min(values_multistep))

# Output results
print("Results:")
for strategy, (final_value, best_value) in results.items():
    print(f"Strategy: {strategy}")
    print(f"  Final function value: {final_value}")
    print(f"  Best function value: {best_value}")
