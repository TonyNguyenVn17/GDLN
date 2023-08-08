import numpy as np
import matplotlib.pyplot as plt
from linear_regression import gradient_descent, mean_squared_error
from data_generation import generate_data
from data_visualization import plot_data_with_regression_line, plot_cost_history

def main():

    #** Data Input - Define parameters for the data 
    num_samples = 100
    num_features = 2
    true_coefficients = np.array([3.0, -1.5])  # Define the true coefficients here
    true_intercept = 2.0
    true_coefficients = np.array([true_intercept] + list(true_coefficients))
    learning_rate = 0.01
    num_iterations = 1000

    # Generate synthetic data
    X, y = generate_data(num_samples, num_features, true_coefficients)

    # Initialize coefficients for linear regression
    initial_coefficients = np.zeros(num_features + 1)  # Additional one for intercept

    # Perform Gradient Descent to estimate coefficients
    history = []
    coefficients = initial_coefficients
    for i in range(num_iterations):
        # Calculate Mean Squared Error (MSE) cost using the current coefficients
        # Use mean_squared_error  with y true y and np.dot(X, coefficients) predicted y
        cost = mean_squared_error(y, np.dot(X, coefficients))  
        history.append(cost)

        # Update coefficients using Gradient Descent
        coefficients = gradient_descent(X, y, coefficients, learning_rate, 1)

    # Print the final estimated coefficients
    print("Estimated Coefficients:", coefficients)


	# Compare true coefficients, estimated coefficients, and reverse-estimated coefficients
    print("True Coefficients:", true_coefficients)
    print("Difference between True and Estimated Coefficients:", true_coefficients - coefficients)


    # Visualize the data with the regression line
    plot_data_with_regression_line(X, y, coefficients)

    # Plot the cost history during Gradient Descent
    plot_cost_history(history)

    # Show the plot windows and wait for them to be closed
    plt.show(block=True)

	
if __name__ == "__main__":
    main()