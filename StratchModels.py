import numpy as np
from sklearn.tree import DecisionTreeClassifier
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
        self.children_left = []
        self.children_right = []
    
    @property
    def n_leaves(self):
        return self.children_left.count(-1) 
    
    def add_node(self):
        self.children_left.append(-1)
        self.children_right.append(-1)
    
    def set_left_child(self, node_id: int, child_id: int):
        self.children_left[node_id] = child_id

    def set_right_child(self, node_id: int, child_id: int):
        self.children_right[node_id] = child_id

    def get_children(self, node_id: int) -> tuple[int]: 
        return self.children_left[node_id], self.children_right[node_id]

    def is_leaf(self, node_id: int) -> bool:
        return self.children_left[node_id] == self.children_right[node_id] #==-1


class DecisionTree:
    def __init__(self, max_depth=None, max_features=None, min_samples_leaf=1, random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.RandomState = check_RandomState(random_state)

        # initialise internal variables
        self.tree_ = BinaryTree() 
        self.n_samples = []
        self.values = []
        self.impurities = []
        self.split_features = []
        self.split_values = []
        self.size = 0 # current node = size - 1
		
    def split_name(self, node_id: int) -> str:
        return self.features[self.split_features[node_id]]

    def fit(self, X, Y):
        if Y.ndim == 1:
            Y = encode_one_hot(Y) # one-hot encoded y variable
    
        # set internal variables
        self.n_features = X.shape[1]
        self.n_classes = Y.shape[1]
        self.features = X.columns
        self.max_depth_ = float('inf') if self.max_depth is None else self.max_depth
        self.n_features_split = self.n_features if self.max_features is None else self.max_features
    
        # initial split which recursively calls itself
        self._split_node(X, Y, 0)  
    
        # set attributes
        self.feature_importances_ = self.impurity_feature_importance()

    def gini_score(self, counts): 
        return 1 - sum([c*c for c in counts])/(sum(counts)*sum(counts))

    def _set_defaults(self, node_id: int, Y):
        val = Y.sum(axis=0)
        self.values.append(val)
        self.impurities.append(self.gini_score(val))
        self.split_features.append(None)
        self.split_values.append(None)
        self.n_samples.append(Y.shape[0])
        self.tree_.add_node()
    
    def _split_node(self, X, Y, depth: int):
        node_id = self.size
        self.size += 1
        self._set_defaults(node_id, Y)
        if self.impurities[node_id] == 0: 
            return
    	
        features = self.RandomState.permutation(self.n_features)[:self.n_features_split]

	#splting 
        best_score = float('inf')
        for i in features:
            best_score = self._find_bettersplit(i, X, Y, node_id, best_score)
        if best_score == float('inf'): # a split was not made
            return 
    
        # children
        if depth < self.max_depth_: 
            x_split = X.values[:, self.split_features[node_id]]
            lhs = np.nonzero(x_split<=self.split_values[node_id])
            rhs = np.nonzero(x_split> self.split_values[node_id])
            self.tree_.set_left_child(node_id, self.size)
            self._split_node(X.iloc[lhs], Y[lhs[0], :], depth+1)
            self.tree_.set_right_child(node_id, self.size)
            self._split_node(X.iloc[rhs], Y[rhs[0], :], depth+1)


    
    def _find_bettersplit(self, var_idx: int, X, Y, node_id: int, best_score: float) -> float:
        X = X.values[:, var_idx] 
        n_samples = self.n_samples[node_id]
    
        order = np.argsort(X)
        X_sort, Y_sort = X[order], Y[order, :]
    
        rhs_count = Y.sum(axis=0)
        lhs_count = np.zeros(rhs_count.shape)
        for i in range(0, n_samples-1):
            xi, yi = X_sort[i], Y_sort[i, :]
            lhs_count += yi;  rhs_count -= yi
            if (xi == X_sort[i+1]) or (sum(lhs_count) < self.min_samples_leaf):
                continue
            if sum(rhs_count) < self.min_samples_leaf:
                break
		    
            curr_score = (self.gini_score(lhs_count) * sum(lhs_count) + self.gini_score(rhs_count) * sum(rhs_count))/n_samples
            if curr_score < best_score:
                best_score = curr_score
                self.split_features[node_id] = var_idx
                self.split_values[node_id]= (xi + X_sort[i+1])/2
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
             # feature_importances  = (decrease in node impurity) * (probability of reaching node ~ proportion of samples)
            feature_importances[spit_feature] += (impurity-score) * (n_samples/total_samples)
    
            return feature_importances/feature_importances.sum()

def parallel_split_node(args):
    decision_tree, X, Y, depth, node_id = args
    decision_tree._split_node(X, Y, depth, node_id)

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
        self.RandomState = check_RandomState(random_state)
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf=min_samples_leaf
        self.sample_size = sample_size
        self.bootstrap = bootstrap
        self.oob_score = oob_score

    # https://www.kite.com/python/answers/how-to-do-one-hot-encoding-with-numpy-in-python(code reference)
    def encode_one_hot(self, data): 
        one_hot = np.zeros((data.size, data.max()+1))
        rows = np.arange(data.size)
        one_hot[rows, data] = 1
        return one_hot

    
    def fit(self, X, Y):
        if Y.ndim == 1:
            Y = self.encode_one_hot(Y) 
    
        self.n_features = X.shape[1]
        self.n_classes = Y.shape[1]
        self.features = X.columns
        n_samples = X.shape[0]
        self.sample_size_ = n_samples if self.sample_size is None else self.sample_size
    
        # create decision trees
        # self.trees = []
        # rng_states = [] # save the random states to regenerate the random indices for the oob_score
        # for i in range(self.n_trees):
        #     rng_states.append(self.RandomState.get_state())
        #     self.trees.append(self._create_tree(X, Y))

        results = Parallel(n_jobs=-1)(
            delayed(self._create_tree)(X, Y, i) for i in range(self.n_trees))

        self.trees, rng_states = zip(*results)      
    
        # set attributes
        self.feature_importances_ = self.impurity_feature_importances()
        if self.oob_score:
            if not (self.bootstrap or (self.sample_size_<n_samples)):
                    warnings.warn("out-of-bag score will not be calculated because bootstrap=False")
            else:
                self.oob_score_ = self.calculate_oob_score(X, Y, rng_states)

    def _create_tree(self, X, Y, tree_idx):
        assert len(X) == len(Y), ""
        n_samples = X.shape[0]
    
        if self.bootstrap: # sample with replacement
            rand_idxs = self.RandomState.randint(0, n_samples, self.sample_size_) 
            # X_, Y_ = X.iloc[rand_idxs, :], Y[rand_idxs] #
        
        elif self.sample_size_ < n_samples: # sample without replacement
            rand_idxs = self.RandomState.permutation(np.arange(n_samples))[:self.sample_size_]  
            # X_, Y_ = X.iloc[rand_idxs, :], Y[rand_idxs]
        else:
            rand_idxs = None
            # X_, Y_ = X.copy(), Y.copy() # do nothing to the data
    
        new_tree =  parallel_run_decision_tree(X.iloc[rand_idxs, :] if rand_idxs is not None else X,
                                Y[rand_idxs] if rand_idxs is not None else Y,
                                max_depth=self.max_depth, 
                                 max_features=self.max_features,
                                 random_state=self.RandomState,
                                 min_samples_leaf=self.min_samples_leaf
                                )
        # new_tree.fit(X_, Y_)
        # new_tree.fit(X.iloc[rand_idxs, :] if rand_idxs is not None else X, Y[rand_idxs] if rand_idxs is not None else Y)
        
        # if (tree_idx + 1) % 10 == 0:
        #     print(f"Tree {tree_idx + 1} built.")
            
        return new_tree, self.RandomState.get_state()

    
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

    def impurity_feature_importances(self) -> np.ndarray:
        feature_importances = np.zeros((self.n_trees, self.n_features))
    
        for i, tree in enumerate(self.trees):
            feature_importances[i, :] = tree.feature_importances_
            
        return np.mean(feature_importances, axis=0)
