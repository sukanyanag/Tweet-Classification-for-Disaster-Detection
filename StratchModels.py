import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



## Cusomt Cross Validation
class CustomCrossValidator:
    def __init__(self, model, cv=5):
        self.model = model
        self.cv = cv
        self.scores = {'accuracy':[], 'precision':[], 'recall':[], 'f1': []}

    def fit_and_evaluate(self, X, target):
        if type(target) != 'numpy.ndarray':
            target = np.array(target)

        skf = StratifiedKFold(n_splits=self.cv)
        for fold, (train, test) in enumerate(skf.split(X, target)):
            self.model.fit(X[train], target[train])

            # Predict on the validation set
            y_pred = self.model.predict(X[test])

            # Calculate the specified scoring metric
            self.scores['accuracy'].append(accuracy_score(target[test], y_pred))
            self.scores['precision'].append(precision_score(target[test], y_pred))
            self.scores['recall'].append(recall_score(target[test], y_pred))
            self.scores['f1'].append(f1_score(target[test], y_pred))

        self.print_average_scores()
        return self.scores

    def print_average_scores(self):
        print('avg accuracy: ', np.mean(self.scores['accuracy']))
        print('avg precision: ', np.mean(self.scores['precision']))
        print('avg recall: ', np.mean(self.scores['recall']))
        print('avg f1: ', np.mean(self.scores['f1']))



## Logistic Regression 
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, num_features):
        self.weights = np.zeros((num_features, 1))
        self.bias = 0

    def fit(self, X, y):
        m, num_features = X.shape
        self.initialize_parameters(num_features)

        for i in range(self.num_iterations):
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            dw = (1/m) * np.dot(X.T, (predictions - y))
            db = (1/m) * np.sum(predictions - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        binary_predictions = (predictions >= 0.5).astype(int)
        return binary_predictions



## Naive Bayes 
class MultinomialNaiveBayes:
    def fit(self, X, y):
        self.n_classes = np.unique(y).shape[0]
        self.n_features = X.shape[1]
        self.class_priors = np.zeros(self.n_classes, dtype=np.float64)
        self.feature_probs = np.zeros((self.n_classes, self.n_features), dtype=np.float64)

        for c in range(self.n_classes):
            X_c = X[y == c]
            self.class_priors[c] = X_c.shape[0] / float(X.shape[0])
            self.feature_probs[c, :] = (X_c.sum(axis=0) + 1) / (X_c.sum() + self.n_features)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        class_probabilities = []
        for c in range(self.n_classes):
            class_probability = np.log(self.class_priors[c]) + np.sum(np.log(self.feature_probs[c, :]) * x)
            class_probabilities.append(class_probability)
        return np.argmax(class_probabilities)


## Random Forest
class RandomForestClassifier:
    def __init__(self, n_trees=100, max_depth=None, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.pool = None

    def fit(self, X, y):
        np.random.seed(self.random_state)

        for _ in range(self.n_trees):
            # Train a decision tree on the random subset
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, y)
            # Append the trained tree to the ensemble
            self.trees.append(tree)

    def predict(self, X):
        # Make predictions using each tree and take a majority vote
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Use the mode function to get the most common prediction for each instance
        ensemble_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

        return ensemble_predictions
