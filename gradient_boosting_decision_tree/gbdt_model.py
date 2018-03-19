from __future__ import division, print_function
import numpy as np
import progressbar

# Import helper functions
from utils import train_test_split, standardize, to_categorical
from utils import mean_squared_error, accuracy_score
from utils.loss_functions import SquareLoss, CrossEntropy, SotfMaxLoss
from decision_tree.decision_tree_model import RegressionTree, ClassificationRegressionTree
from utils.misc import bar_widgets


class GBDT(object):
    """Super class of GradientBoostingClassifier and GradientBoostinRegressor.
    Uses a collection of regression trees that trains on predicting the gradient
    of the loss function.

    Parameters:
    -----------
    n_estimators: int
        树的数量
        The number of classification trees that are used.
    learning_rate: float
        梯度下降的学习率
        The step length that will be taken when following the negative gradient during
        training.
    min_samples_split: int
        每棵子树的节点的最小数目（小于后不继续切割）
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        每颗子树的最小纯度（小于后不继续切割）
        The minimum impurity required to split the tree further.
    max_depth: int
        每颗子树的最大层数（大于后不继续切割）
        The maximum depth of a tree.
    regression: boolean
        是否为回归问题
        True or false depending on if we're doing regression or classification.
    """

    def __init__(self, n_estimators, learning_rate, min_samples_split,
                 min_impurity, max_depth, regression):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.regression = regression

        # 进度条 processbar
        self.bar = progressbar.ProgressBar(widgets=bar_widgets)

        self.loss = SquareLoss()
        if not self.regression:
            self.loss = SotfMaxLoss()

        # 分类问题也使用回归树，利用残差去学习概率
        # Change 2: 若为K分类问题，建立K*n_estimators棵ClassificationRegressionTree

        if self.regression:
            self.trees = []
            for i in range(self.n_estimators):
                self.trees.append(RegressionTree(min_samples_split=self.min_samples_split,
                                                 min_impurity=self.min_impurity,
                                                 max_depth=self.max_depth))

    def fit(self, X, y):
        # 让第一棵树去拟合模型
        self.trees[0].fit(X, y)
        y_pred = self.trees[0].predict(X)
        for i in self.bar(range(1, self.n_estimators)):
            gradient = self.loss.gradient(y, y_pred) # y_pred = p if GBDTClassifier
            self.trees[i].fit(X, gradient)
            y_pred -= np.multiply(self.learning_rate, self.trees[i].predict(X))

    def predict(self, X):
        y_pred = self.trees[0].predict(X)
        for i in range(1, self.n_estimators):
            y_pred -= np.multiply(self.learning_rate, self.trees[i].predict(X))

        if not self.regression:
            # Turn into probability distribution
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            # Set label to the value that maximizes probability
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred


class GBDTRegressor(GBDT):
    def __init__(self, n_estimators=200, learning_rate=0.5, min_samples_split=2,
                 min_var_red=1e-7, max_depth=4, debug=False):
        super(GBDTRegressor, self).__init__(n_estimators=n_estimators,
                                            learning_rate=learning_rate,
                                            min_samples_split=min_samples_split,
                                            min_impurity=min_var_red,
                                            max_depth=max_depth,
                                            regression=True)


class GBDTClassifier(GBDT):
    def __init__(self, n_classes, n_estimators=200, learning_rate=.5, min_samples_split=2,
                 min_info_gain=1e-7, max_depth=2, debug=False):
        self.n_classes = n_classes
        super(GBDTClassifier, self).__init__(n_estimators=n_estimators,
                                             learning_rate=learning_rate,
                                             min_samples_split=min_samples_split,
                                             min_impurity=min_info_gain,
                                             max_depth=max_depth,
                                             regression=False)

    def fit(self, X, y):

        # Change 3: 重载fit函数，训练所有的树
        y = to_categorical(y) # onehot for y
        K = self.n_classes
        M = self.n_estimators
        N = X.shape[0]

        # Initialize trees
        self.trees = []
        for i in range(M):
            trees_k = []
            for j in range(K):
                trees_k.append(ClassificationRegressionTree(n_classes=K,
                                                            min_samples_split=self.min_samples_split,
                                                            min_impurity=self.min_impurity,
                                                            max_depth=self.max_depth))
            self.trees.append(trees_k)

        p = np.zeros([N, K])
        for m in range(M):
            print("Round ", m)
            p = np.exp(p) / np.expand_dims(np.exp(p).dot(np.ones(K)), axis=1)
            for k in range(K):
                gradient = self.loss.gradient(y[:,k], p[:,k])
                self.trees[m][k].fit(X, gradient)
                a = self.trees[m][k].predict(X)
                p[:,k] = p[:,k] + np.multiply(a, self.learning_rate)


    def predict(self, X):
        N = X.shape[0]
        K = self.n_classes

        y_pred = []
        for i in range(N):
            y_pred.append(np.zeros(K))
        y_pred = np.array(y_pred)
        print(y_pred.shape)

        for k in range(K):
            y_pred[:,k] = self.trees[0][k].predict(X)
        for i in range(1, self.n_estimators):
            for k in range(K):
                y_pred[:,k] += np.multiply(self.learning_rate, self.trees[i][k].predict(X))

        if not self.regression:
            # Turn into probability distribution
            y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
            # Set label to the value that maximizes probability
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred