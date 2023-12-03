import numpy as np
from scipy.optimize import minimize

class LogisticRegression:
    def __init__(self, fit_intercept=True):
        """
        Initialize the logistic regression model.

        Parameters:
        - fit_intercept (bool): Whether to add an intercept term to the features.
        """
        self.fit_intercept = fit_intercept
        self.theta = None

    def sigmoid(self, z):
        """
        Compute the sigmoid (logistic) function.

        Parameters:
        - z (numpy.ndarray): Input values.

        Returns:
        - numpy.ndarray: Sigmoid of input values.
        """
        return 1 / (1 + np.exp(-z))

    def add_intercept(self, X):
        """
        Add an intercept term to the features.

        Parameters:
        - X (numpy.ndarray): Input features.

        Returns:
        - numpy.ndarray: Features with an added intercept term.
        """
        return np.insert(X, 0, 1, axis=1) if self.fit_intercept else X

    def cost_function(self, theta, X, y):
        """
        Compute the cost function for logistic regression.

        Parameters:
        - theta (numpy.ndarray): Model parameters.
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): True labels.

        Returns:
        - float: Cost value.
        """
        m = len(y)
        h = self.sigmoid(X @ theta)
        cost = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost

    def gradient(self, theta, X, y):
        """
        Compute the gradient of the cost function.

        Parameters:
        - theta (numpy.ndarray): Model parameters.
        - X (numpy.ndarray): Input features.
        - y (numpy.ndarray): True labels.

        Returns:
        - numpy.ndarray: Gradient vector.
        """
        m = len(y)
        h = self.sigmoid(X @ theta)
        grad = (1 / m) * X.T @ (h - y)
        return grad

    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.

        Parameters:
        - X (numpy.ndarray): Training features.
        - y (numpy.ndarray): Training labels.
        """
        if self.fit_intercept:
            X = self.add_intercept(X)

        initial_theta = np.zeros(X.shape[1])
        result = minimize(self.cost_function, initial_theta, args=(X, y), jac=self.gradient, method='L-BFGS-B')

        if result.success:
            self.theta = result.x
        else:
            raise Exception("Optimization failed: " + result.message)

    def predict_proba(self, X):
        """
        Predict probabilities for the given input features.

        Parameters:
        - X (numpy.ndarray): Input features.

        Returns:
        - numpy.ndarray: Predicted probabilities.
        """
        if self.fit_intercept:
            X = self.add_intercept(X)

        return self.sigmoid(X @ self.theta)

    def predict(self, X, threshold=0.5):
        """
        Make binary predictions based on the given input features.

        Parameters:
        - X (numpy.ndarray): Input features.
        - threshold (float): Threshold for binary classification.

        Returns:
        - numpy.ndarray: Binary predictions (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)