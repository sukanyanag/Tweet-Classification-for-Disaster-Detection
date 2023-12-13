import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from joblib import Parallel, delayed
import concurrent.futures



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


## Random forest (multiple decision trees)


# Decision Tree 
class BinaryTree():
    def __init__(self):
        self.left_children = []
        self.right_children = []
    
    @property
    def num_leaves(self):
        return self.left_children.count(-1) 
    
    def add_node(self):
        self.left_children.append(-1)
        self.right_children.append(-1)
    
    def set_left_child(self, id_node: int, child_id: int):
        self.left_children[id_node] = child_id

    def set_right_child(self, id_node: int, child_id: int):
        self.right_children[id_node] = child_id

    def get_children(self, id_node: int) -> tuple[int]: 
        return self.left_children[id_node], self.right_children[id_node]

    def is_leaf(self, id_node: int) -> bool:
        return self.left_children[id_node] == self.right_children[id_node]


class DecisionTree:
    def __init__(self, max_depth=None, max_features=None, min_samples_leaf=1, random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.random_state = check_RandomState(random_state)

        self.tree_ = BinaryTree() 
        self.n_samples = []
        self.values = []
        self.impurities = []
        self.split_features = []
        self.split_values = []
        self.size = 0 
		
    def split_name(self, id_node: int) -> str:
        return self.features[self.split_features[id_node]]

    def fit(self, X, Y):
        if Y.ndim == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable
    
        # set internal variables
        self.n_features = X.shape[1]
        self.n_classes = Y.shape[1]
        self.features = np.sqrt(X.shape[1])
        self.max_depth_ = float('inf') if self.max_depth is None else self.max_depth
        self.n_features_split = self.n_features if self.max_features is None else self.max_features
    
        # initial split which recursively calls itself
        self._split_node(X, Y, 0)  
    
        # set attributes
        self.feature_importances_ = self.impurity_feature_importance()

    def gini_score(self, counts):
        total_samples = sum(counts)
        if total_samples == 0:
            return 0 
    
        probabilities = counts / total_samples
        gini = 1 - np.sum(probabilities**2)
        return gini

    def _set_defaults(self, id_node: int, Y):
        val = Y.sum(axis=0)
        self.values.append(val)
        self.impurities.append(self.gini_score(val))
        self.split_features.append(None)
        self.split_values.append(None)
        self.n_samples.append(Y.shape[0])
        self.tree_.add_node()
    
    def _split_node(self, X, Y, depth: int):
        id_node = self.size
        self.size += 1
        self._set_defaults(id_node, Y)
        if self.impurities[id_node] == 0: 
            return
    	
        features = self.random_state.permutation(self.n_features)[:self.n_features_split]
    
        best_score = float('inf')
        for i in features:
            best_score = self._find_bettersplit(i, X, Y, id_node, best_score)
        if best_score == float('inf'): # a split was not made
            return 
    
        if depth < self.max_depth_: 
            x_split = X.values[:, self.split_features[id_node]]
            lhs = np.nonzero(x_split<=self.split_values[id_node])
            rhs = np.nonzero(x_split> self.split_values[id_node])
            self.tree_.set_left_child(id_node, self.size)
            self._split_node(X.iloc[lhs], Y[lhs[0], :], depth+1)
            self.tree_.set_right_child(id_node, self.size)
            self._split_node(X.iloc[rhs], Y[rhs[0], :], depth+1)


    
    def _find_bettersplit(self, var_idx: int, X, Y, id_node: int, best_score: float) -> float:
        x_values = X.values[:, var_idx]
        n_samples = self.n_samples[id_node]
    
        order = np.argsort(x_values)
        x_sort, y_sort = x_values[order], Y[order, :]
    
        rhs_count = np.sum(Y, axis=0)
        lhs_count = np.zeros(rhs_count.shape)
    
        for i in range(0, n_samples - 1):
            xi, yi = x_sort[i], y_sort[i, :]
            lhs_count += yi
            rhs_count -= yi
    
            if (xi == x_sort[i + 1]) or (np.sum(lhs_count) < self.min_samples_leaf):
                continue
            if np.sum(rhs_count) < self.min_samples_leaf:
                break
    
            curr_score = (self.gini_score(lhs_count) * np.sum(lhs_count) +
                          self.gini_score(rhs_count) * np.sum(rhs_count)) / n_samples
    
            if curr_score < best_score:
                best_score = curr_score
                self.split_features[id_node] = var_idx
                self.split_values[id_node] = (xi + x_sort[i + 1]) / 2
    
        return best_score

    def _predict_batch(self, X, node=0):
        if self.tree_.is_leaf(node):
            return self.values[node]
        if len(X) == 0:
            return np.empty((0, self.n_classes))
        left, right = self.tree_.get_children(node)
    
        lhs = X[:, self.split_features[node]] <= self.split_values[node]
        rhs = X[:, self.split_features[node]] >  self.split_values[node]
    
        probs = np.zeros((X.shape[0], self.n_classes))
        probs[lhs] = self._predict_batch(X[lhs], node=left)
        probs[rhs] = self._predict_batch(X[rhs], node=right)
        return probs
    
    def predict_prob(self, X):
        probs = self._predict_batch(X.values)
        probs /= np.sum(probs, axis=1)[:, None] # normalise along each row (sample)
        return probs
    
    def predict(self, X):
        probs = self.predict_prob(X)
        return np.nanargmax(probs, axis=1)


    def impurity_feature_importance(self):
        feature_importances = np.zeros(self.n_features)
        total_samples = self.n_samples[0]
        for node in range(len(self.impurities)):
            if self.tree_.is_leaf(node):
                continue 
            spit_feature = self.split_features[node]
            impurity = self.impurities[node]
            n_samples = self.n_samples[node]
            # calculate score
            left, right = self.tree_.get_children(node)
            lhs_gini = self.impurities[left]
            rhs_gini = self.impurities[right]
            lhs_count = self.n_samples[left]
            rhs_count = self.n_samples[right]
            score = (lhs_gini * lhs_count + rhs_gini * rhs_count)/n_samples
            feature_importances[spit_feature] += (impurity-score) * (n_samples/total_samples)
    
            return feature_importances/feature_importances.sum()

def parallel_split_node(args):
    decision_tree, X, Y, depth, id_node = args
    decision_tree._split_node(X, Y, depth, id_node)

def parallel_fit(decision_tree, X, Y):
    if Y.ndim == 1:
        Y = encode_one_hot(Y)

    decision_tree.n_features = X.shape[1]
    decision_tree.n_classes = Y.shape[1]
    decision_tree.features = X.columns
    decision_tree.max_depth_ = float('inf') if decision_tree.max_depth is None else decision_tree.max_depth
    decision_tree.n_features_split = decision_tree.n_features if decision_tree.max_features is None else decision_tree.max_features

    decision_tree._split_node(X, Y, 0)

    decision_tree.feature_importances_ = decision_tree.impurity_feature_importance()

def parallel_run_decision_tree(X, Y, max_depth=None, max_features=None, min_samples_leaf=1, random_state=None, max_workers=None):
    decision_tree = DecisionTree(
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Split the root node in parallel
        executor.submit(parallel_fit, decision_tree, X, Y)

    return decision_tree

import numpy as np
import pandas as pd
import warnings


def check_RandomState(random_state):
    """ Parse different input types for the random state"""
    if  random_state is None: 
        rng = np.random.RandomState() 
    elif isinstance(random_state, int): 
        # seed the random state with this integer
        rng = np.random.RandomState(random_state) 
    elif isinstance(random_state, np.random.RandomState):
        rng = random_state
    else:
        raise ValueError ("improper type \'%s\' for random_state parameter" % type(random_state))
    return rng


class RandomForestClassifier:
    
    def __init__(self, n_trees=100, random_state=None, max_depth=None,  
                 max_features=None, min_samples_leaf=1, sample_size=None, 
                 bootstrap=True,  oob_score=False):
        self.n_trees = n_trees
        self..random_state = check_RandomState(random_state)
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf=min_samples_leaf
        self.sample_size = sample_size
        self.bootstrap = bootstrap
        self.oob_score = oob_score

    
    def encode_one_hot(self, data): # note: pd.get_dummies(df) does the same
        # https://www.kite.com/python/answers/how-to-do-one-hot-encoding-with-numpy-in-python
        one_hot = np.zeros((data.size, data.max()+1))
        rows = np.arange(data.size)
        one_hot[rows, data] = 1
        return one_hot

    
    def fit(self, X, Y):
        if Y.ndim == 1:
            Y = self.encode_one_hot(Y) # one-hot encoded y variable
    
        # set internal variables
        self.n_features = X.shape[1]
        self.n_classes = Y.shape[1]
        self.features = X.columns
        n_samples = X.shape[0]
        self.sample_size_ = n_samples if self.sample_size is None else self.sample_size
    

        results = Parallel(n_jobs=-1)(
            delayed(self._create_tree)(X, Y, i) for i in range(self.n_trees))

        self.trees, rng_states = zip(*results)      
    
        self.feature_importances_ = self._calculate_impurity_feature_importances()
    
        if self.oob_score:
            if not (self.bootstrap or (self.sample_size_<n_samples)):
                    warnings.warn("out-of-bag score will not be calculated because bootstrap=False")
            else:
                self.oob_score_ = self.calculate_oob_score(X, Y, rng_states)

    def _create_tree(self, X, Y, tree_idx):
        assert len(X) == len(Y), ""
        n_samples = X.shape[0]
    
        if self.bootstrap: 
            rand_idxs = self..random_state.randint(0, n_samples, self.sample_size_) 
        
        elif self.sample_size_ < n_samples: # sample without replacement
            rand_idxs = self..random_state.permutation(np.arange(n_samples))[:self.sample_size_]  
        else:
            rand_idxs = None
    
        new_tree =  parallel_run_decision_tree(X.iloc[rand_idxs, :] if rand_idxs is not None else X,
                                Y[rand_idxs] if rand_idxs is not None else Y,
                                max_depth=self.max_depth, 
                                 max_features=self.max_features,
                                 random_state=self..random_state,
                                 min_samples_leaf=self.min_samples_leaf
                                )
            
        return new_tree, self..random_state.get_state()

    
    def predict(self, X) -> np.ndarray:
        probs = np.sum([t.predict_prob(X) for t in self.trees], axis=0)
        return np.nanargmax(probs, axis=1)
    
    def score(self, X, y) -> float:
        y_pred = self.predict(X)
        return np.mean(y_pred==y)

    def calculate_oob_score(self, X, Y, rng_states):
        n_samples = X.shape[0]
        oob_prob = np.zeros(Y.shape)
        oob_count = np.zeros(n_samples)
        rng = np.random.RandomState()
        # regenerate random samples using the saved random states
        for i, state in enumerate(rng_states):
            rng.set_state(state) 
            if self.bootstrap: # sample with replacement
                rand_idxs = rng.randint(0, n_samples, self.sample_size_)
            else: #self.sample_size_ < n_samples, # sample without replacement
                rand_idxs = rng.permutation(np.arange(n_samples))[:self.sample_size_]
            row_oob = np.setxor1d(np.arange(n_samples), rand_idxs)
            oob_prob[row_oob, :] += self.trees[i].predict_prob(X.iloc[row_oob])
            oob_count[row_oob] += 1
        valid = oob_count > 0 
        oob_prob = oob_prob[valid, :]
        oob_count = oob_count[valid][:, np.newaxis] # transform to column vector for broadcasting during the division
        y_test    =  np.argmax(Y[valid], axis=1)
        y_pred = np.argmax(oob_prob/oob_count, axis=1)
        return np.mean(y_pred==y_test)

    def _calculate_impurity_feature_importances(self) -> np.ndarray:
        feature_importances = np.zeros((self.n_trees, self.n_features))
    
        for i, tree in enumerate(self.trees):
            feature_importances[i, :] = tree.feature_importances_
            
        return np.mean(feature_importances, axis=0)
