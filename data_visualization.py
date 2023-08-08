import matplotlib.pyplot as plt
import numpy as np

def plot_data_with_regression_line(X, y, coefficients=None, history=None):
    """
    Plots the data points along with the regression line.

    Parameters:
        X (numpy array): Array of shape (num_samples, num_features) representing the features.
        y (numpy array): Array of shape (num_samples,) representing the dependent variable.
        coefficients (array-like): Coefficients of the regression line (optional).
        history (list): List of cost values during Gradient Descent (optional).
    """
    plt.scatter(X[:, 1], y, marker='o', label='Data Points')
    
    if coefficients is not None:
        x_values = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
        y_values = coefficients[0] + coefficients[1] * x_values
        plt.plot(x_values, y_values, color='red', label='Regression Line')

    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.title('Data with Regression Line')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cost_history(history):
    """
    Plots the cost function's history during Gradient Descent.

    Parameters:
        history (list): List of cost values during Gradient Descent.
    """
    plt.plot(range(len(history)), history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost History during Gradient Descent')
    plt.grid(True)
    plt.show()
