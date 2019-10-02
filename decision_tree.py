from pre_processing import get_data_for_model
from utils import *
from sklearn.tree import DecisionTreeClassifier
from utils import cross_validation
from BinaryTree import BinaryTree
import numpy as np
import time


# https://dataaspirant.com/2017/01/30/how-decision-tree-algorithm-works/
# https://www.geeksforgeeks.org/decision-tree-introduction-example/
# https://en.wikipedia.org/wiki/Decision_tree
# https://www.youtube.com/watch?v=Qdi0GBWrDO8
# https://medium.com/deep-math-machine-learning-ai/chapter-4-decision-trees-algorithms-b93975f7a1f1
# https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
# http://www.csun.edu/~twang/595DM/Slides/Week4.pdf
# https://learning.oreilly.com/library/view/data-mining-concepts/9780123814791/xhtml/ST0025_CHP008.html#ST0025_CHP008
class DecisionTree:
    "Decision Tree algorithm"
    tree = None

    def create_training_label(self, training, label, index):
        has_word_index = np.nonzero(training[:, index])[0]
        has_not_word_index = np.where(training[:, 0] == 0)[0]
        has_word_training = training[has_word_index]
        has_word_label = label[has_word_index]
        has_not_word_training = training[has_not_word_index]
        has_not_word_label = label[has_not_word_index]
        return has_word_training, has_word_label, has_not_word_training, has_not_word_label

    def build_tree(self, training, label, removed_features=None):
        if removed_features is None:
            removed_features = []
        tree = BinaryTree()
        major_class = sum(label)
        if major_class == 0:
            tree.value = TRUE_NEWS
            tree.frequency = label.shape[0]
            return tree
        elif major_class == label.shape[0]:
            tree.value = FAKE_NEWS
            tree.frequency = label.shape[0]
            return tree
        elif len(removed_features) == training.shape[1] - 1:
            if major_class >= label.size / 2:
                tree.value = TRUE_NEWS
                tree.frequency = major_class
            else:
                tree.value = FAKE_NEWS
                tree.frequency = label.shape[0] - major_class
            return tree
        gini_value, word = calculate_gini(training, label, removed_features)
        hwt, hwl, hnwt, hnwl = self.create_training_label(training, label, word)
        removed_features.append(word)
        if len(hwt) <= 2 or len(hnwt) <= 100:
            if major_class >= len(label) / 2:
                tree.value = TRUE_NEWS
                tree.frequency = major_class
            else:
                tree.value = FAKE_NEWS
                tree.frequency = label.shape[0] - major_class
            return tree
        tree.value = word
        tree.gini = gini_value
        tree.frequency = len(hwt) + len(hnwt)
        tree.insertHasWord(self.build_tree(hwt, hwl, removed_features))
        tree.insertHasNoWord(self.build_tree(hnwt, hnwl, removed_features))
        return tree

    def fit(self, training, label):
        init = time.time()
        self.tree = self.build_tree(np.array(training), np.array(label))
        end = time.time()
        print(end - init)

    def predict(self, training):
        predictions = []
        for case in training:
            predictions.append(self.tree.find(case))
        return predictions


X, y = get_data_for_model()
print("start")
init = time.time()
# clf_gini = DecisionTreeClassifier(criterion="gini")
cross_validation(X, y, DecisionTree())
end = time.time() - init
print(end)
