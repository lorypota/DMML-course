from exercise1_gradient_descent import *

def main():
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



if __name__ == "__main__":
    main()