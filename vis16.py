import numpy as np
from scipy.optimize import minimize

# Example initial data
u = np.array([529.137, 528.134, 527.597])

v = np.array([1, 4, 7])
X = np.array([8.084, 8.068, 8.058])
Y = np.array([7.319, 6.770, 6.213])
Z = np.array([-4.692, -4.889, -4.981])


# Define the objective function
def objective(params):
    u_new = params[:len(u)]
    v_new = params[len(u):]

    # Simple linear model for demonstration purposes
    X_pred = 0.0168 * u_new - 0.7865
    Y_pred = -0.1843 * v_new + 7.5047

    # Calculate the residuals for X and Y predictions
    residual_X = np.sum((X - X_pred) ** 2)
    residual_Y = np.sum((Y - Y_pred) ** 2)

    # Combine residuals (add Z optimization if needed)
    return residual_X + residual_Y


# Initial guess
initial_params = np.concatenate([u, v])

# Perform optimization
result = minimize(objective, initial_params, method='L-BFGS-B')

# Extract optimized u and v
optimized_params = result.x
u_optimized = optimized_params[:len(u)]
v_optimized = optimized_params[len(u):]

print(544-u, v)
print(544-u_optimized, v_optimized)
