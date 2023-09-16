import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self, degree=1, learning_rate=0.01, num_iterations=1000, regularization=0.01, batch_size=32, convergence_threshold=1e-4):
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.batch_size = batch_size
        self.convergence_threshold = convergence_threshold
        
        self.bias = None
        self.weights = None

    def generate_polynomial_features(self, X):
        X_poly = X.copy()
        for d in range(2, self.degree + 1):
            X_poly = np.concatenate((X_poly, X ** d), axis=1)
        return X_poly

    def fit(self, X, y):
        X_poly = self.generate_polynomial_features(X)
        m, n = X_poly.shape
        self.weights = np.random.randn(n)
        self.bias = 0

        for _ in range(self.num_iterations):
            for i in range(0, m, self.batch_size):
                # Create mini-batch
                X_batch = X_poly[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                # Compute predictions for the mini-batch
                y_pred = sigmoid(np.dot(X_batch, self.weights) + self.bias)

                # Compute gradients for the mini-batch
                dw = (1 / X_batch.shape[0]) * np.dot(X_batch.T, (y_pred - y_batch)) + (2 * self.regularization * self.weights)
                db = (1 / X_batch.shape[0]) * np.sum(y_pred - y_batch)

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model not trained. Call fit() first.")
        X_poly = self.generate_polynomial_features(X)
        y_pred = sigmoid(np.dot(X_poly, self.weights) + self.bias)
        return y_pred


# --- Some utility functions

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true
    return correct_predictions.mean()


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise

    Hint: highly related to cross-entropy loss 

    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.

    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))
