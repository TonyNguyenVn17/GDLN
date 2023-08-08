import numpy as np

def gradient_descent(x, y, initial_coefficients, learning_rate, num_iterations):
    """
    Perform Gradient Descent to estimate the coefficients for Linear Regression.

    Parameters:
        x (numpy array): Input features of shape (num_samples, num_features).
        y (numpy array): Target values of shape (num_samples,).
        initial_coefficients (numpy array): Initial coefficients (including intercept) of shape (num_features + 1,).
        learning_rate (float): Learning rate for the Gradient Descent.
        num_iterations (int): Number of iterations for the Gradient Descent.

    Returns:
        numpy array: Estimated coefficients (including intercept) of shape (num_features + 1,).
    """
    num_samples, num_features = x.shape
    coefficients = np.copy(initial_coefficients)

    for _ in range(num_iterations):
        predictions = np.dot(x, coefficients)
        errors = predictions - y
        gradient = np.dot(x.T, errors) / num_samples
        coefficients -= learning_rate * gradient

    return coefficients


def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) for the predicted values.

    Parameters:
        y_true (numpy array): Array of shape (num_samples,) representing the true target values.
        y_pred (numpy array): Array of shape (num_samples,) representing the predicted target values.

    Returns:
        mse (float): The Mean Squared Error between y_true and y_pred.
    """
    mse = np.mean((y_true - y_pred) ** 2) # equivilant too (1 / n) * Σ(y_true - y_pred)^2 in math
    return mse

