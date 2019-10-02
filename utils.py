from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from dataset_reader import *
from nltk.corpus import stopwords
from string import punctuation
import re
import numpy as np
from numba import jit

META_DESCRIPTION = ['author', 'link', 'category', 'date of publication', 'number of tokens',
                    'number of words without punctuation',
                    'number of types', 'number of links inside the news', 'number of words in upper case',
                    'number of verbs', 'number of subjuntive and imperative verbs',
                    'number of nouns', 'number of adjectives', 'number of adverbs',
                    'number of modal verbs (mainly auxiliary verbs)',
                    'number of singular first and second personal pronouns', 'number of plural first personal pronouns',
                    'number of pronouns', 'pausality', 'number of characters', 'average sentence length',
                    'average word length',
                    'percentage of news with speeling errors', 'emotiveness', 'diversity']

meta_true_info, meta_false_info = meta_information()
full_true_info, full_false_info = full_text()
norm_true_info, norm_false_info = norm_text()
stop_words = set(stopwords.words('portuguese') + list(punctuation))
FAKE_NEWS = 1
TRUE_NEWS = 0


def word_filter(word):
    word = re.sub(r'[^\w\s]', '', word).lower()
    if word not in stop_words and word.isalpha():
        return word
    else:
        return ''


def split_data(bag_of_words, label):
    return train_test_split(bag_of_words, label, test_size=0.33, random_state=42)


def values(data, index):
    resp = []
    for i in index:
        resp.append(data[i])
    return resp


def accuracy(confusion_matrix):
    return (confusion_matrix[0][0] + confusion_matrix[1][1]) / (
            confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])


def precision(confusion_matrix):
    return confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])


def recall(confusion_matrix):
    return confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])


def f1_score(precision_value, recall_value):
    return 2 * (precision_value * recall_value) / (precision_value + recall_value)


def metrics(precision_1, recall_1, f1_score_1, confusion_matrix_2):
    precision_2 = precision(confusion_matrix_2)
    recall_2 = recall(confusion_matrix_2)
    f1_score_2 = f1_score(precision_2, recall_2)
    return (precision_1 + precision_2) / 2, (recall_1 + recall_2) / 2, (f1_score_1 + f1_score_2) / 2


def cross_validation(X, y, model):
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)
    precision_1 = 0
    recall_1 = 0
    f1_score_1 = 0
    for train_index, test_index in cv.split(X, y):
        X_train, X_test, y_train, y_test = \
            values(X, train_index), values(X, test_index), values(y, train_index), values(y, test_index)
        print("trainning")
        model.fit(X_train, y_train)
        print("predicting")
        predictions = model.predict(X_test)
        matrix = confusion_matrix(y_test, predictions)
        if precision_1 == 0 and recall_1 == 0 and f1_score_1 == 0:
            precision_1 = precision(matrix)
            recall_1 = recall(matrix)
            f1_score_1 = f1_score(precision_1, recall_1)
        else:
            precision_1, recall_1, f1_score_1 = metrics(precision_1, recall_1, f1_score_1, matrix)
            print("precision: " + str(np.round(precision_1, 2)))
            print("recall: " + str(np.round(recall_1, 2)))
            print("f1 score: " + str(np.round(f1_score_1, 2)))
    print("precision: " + str(np.round(precision_1, 2)))
    print("recall: " + str(np.round(recall_1, 2)))
    print("f1 score: " + str(np.round(f1_score_1, 2)))


# @jit(nopython=True)
def calculate_gini(training, label, removed_features):
    gini_index = 999
    index = -1
    for feature in range(len(training[0])):
        if feature in removed_features:
            continue
        has_word_aux = np.nonzero(training[:, feature])[0]
        has_word_positive = np.count_nonzero(label[has_word_aux]) + 1
        has_word_aux = len(has_word_aux)
        has_not_word_aux = np.where(training[:, 0] == 0)[0]
        has_not_word_positive = np.count_nonzero(label[has_not_word_aux]) + 1
        has_not_word_aux = len(has_not_word_aux)
        has_not_word_negative = has_not_word_aux - has_not_word_positive + 1
        has_word_negative = has_word_aux - has_word_positive + 1
        has_word_total = has_word_positive + has_word_negative
        has_not_word_total = has_not_word_positive + has_not_word_negative
        total = has_word_total + has_not_word_total
        gini_has_word = 1 - ((has_word_positive / has_word_total) ** 2 + (has_word_negative / has_word_total) ** 2)
        gini_has_not_word = 1 - ((has_not_word_positive / has_not_word_total) ** 2 +
        (has_not_word_negative / has_not_word_total) ** 2)
        gini = (has_word_total / total) * gini_has_word + (has_not_word_total / total) * gini_has_not_word
        if gini_index > gini:
            gini_index = gini
            index = feature
    return gini_index, index
