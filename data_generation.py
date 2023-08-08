# data_generation.py
import numpy as np

# Data generation function
def generate_data(num_samples, num_features, true_coefficients):
    # Generate random features with values between 0 and 1
    X = np.random.rand(num_samples, num_features)

    # Add a column of ones for the bias term (intercept)
    X = np.hstack((np.ones((num_samples, 1)), X))

    # Calculate the dependent variable using true coefficients and add Gaussian noise
    # add Gausian noise here
    noise = np.random.normal(0, 0.1, size=num_samples)
    y = np.dot(X, true_coefficients) + noise

    return X, y
