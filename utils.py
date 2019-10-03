from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from dataset_reader import *
from nltk.corpus import stopwords
from string import punctuation
import re
import numpy as np

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


# negative = True News
# positive = Fake News

def accuracy(confusion_matrix):
    return (confusion_matrix[0][0] + confusion_matrix[1][1]) / (
            confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])


def precision(confusion_matrix):
    precision_negative = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    precision_positive = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
    return precision_negative, precision_positive


def recall(confusion_matrix):
    recall_negative = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    recall_positive = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0])
    return recall_negative, recall_positive


def f1_score(precision_value, recall_value):
    return 2 * (precision_value * recall_value) / (precision_value + recall_value)


def metrics(confusion_matrix):
    precision_true_news, precision_fake_news = precision(confusion_matrix)
    recall_true_news, recall_fake_news = recall(confusion_matrix)
    f1_score_true_news = f1_score(precision_true_news, recall_true_news)
    f1_score_fake_news = f1_score(precision_fake_news, recall_fake_news)
    accuracy_news = accuracy(confusion_matrix)
    print("True News Metrics")
    print("Precision: " + str(round(precision_true_news, 2)))
    print("Recall: " + str(round(recall_true_news, 2)))
    print("F1 Score: " + str(round(f1_score_true_news, 2)))
    print("Fake News Metrics")
    print("Precision: " + str(round(precision_fake_news, 2)))
    print("Recall: " + str(round(recall_fake_news, 2)))
    print("F1 Score: " + str(round(f1_score_fake_news, 2)))
    print("Model Metrics")
    print("Model Accuracy: " + str(round(accuracy_news, 2)))


def cross_validation(X, y, model):
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)
    k = 0
    matrix = np.zeros((2, 2))
    training_time_total =0
    print(model.__doc__[0:30])
    for train_index, test_index in cv.split(X, y):
        k += 1
        X_train, X_test, y_train, y_test = \
            values(X, train_index), values(X, test_index), values(y, train_index), values(y, test_index)
        print("Trainning Fold: " + str(k))
        init = time.time()
        model.fit(X_train, y_train)
        end = time.time() - init
        training_time_total +=end
        print("Time spent training: " + str(round(end, 2)))
        print("Predicting Fold: " + str(k))
        init = time.time()
        predictions = model.predict(X_test)
        end = time.time() - init
        print("Time spent predicting: " + str(round(end, 2)))
        print()
        matrix += confusion_matrix(y_test, predictions)
    metrics(matrix)
    print()
    print("Training time spent: "+str(round(training_time_total,2)))
    print("Training time average per fold: "+str(round(training_time_total/5,2)))
