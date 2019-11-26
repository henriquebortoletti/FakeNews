import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from pre_processing import *


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


def extract_metrics(model, name):
    for i in [1, 2, 3]:
        for j in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            f = open(name + ".txt", "w+")
            time, accur = cross_validation(j, i, model)
            f.write(
                "grams: " + str(i) + " bow_percentage: " + str(j) + " time spent: " + str(time) + " accuracy: " + str(
                    accur)+"\r\n")
            f.close()



def cross_validation(bow_percentage_size, n_grams, model):
    print(model.__doc__[0:30])
    X, y = get_data_for_model(bow_percentage_size, n_grams)
    cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=False)
    k = 0
    matrix = np.zeros((2, 2))
    training_time_total = 0
    print("N-Grams: " + str(n_grams))
    print("Bow percentage: " + str(bow_percentage_size))
    print()
    for train_index, test_index in cv.split(X, y):
        k += 1
        X_train, X_test, y_train, y_test = \
            values(X, train_index), values(X, test_index), values(y, train_index), values(y, test_index)
        print("Trainning Fold: " + str(k))
        init = time.time()
        model.fit(X_train, y_train)
        end = time.time() - init
        training_time_total += end
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
    print("Training time spent: " + str(round(training_time_total, 2)))
    print("Training time average per fold: " + str(round(training_time_total / 5, 2)))
    return training_time_total, accuracy(matrix)
