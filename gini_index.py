import numpy as np


def calculate_gini(training_aux, label, removed_features):
    gini_index = 999 #valor inicial default bem grande para que no começo selecione o primeiro
    index = 0
    for feature in range(training_aux.shape[1]):
        if feature in removed_features:
            continue
        has_word_aux = np.nonzero(training_aux[:, feature])[0]
        has_word_positive = np.count_nonzero(label[has_word_aux])
        has_word_aux = has_word_aux.shape[0]

        has_not_word_aux = np.where(training_aux[:, feature] == 0)[0]
        has_not_word_positive = np.count_nonzero(label[has_not_word_aux])
        has_not_word_aux = has_not_word_aux.shape[0]

        has_not_word_negative = has_not_word_aux - has_not_word_positive
        has_word_negative = has_word_aux - has_word_positive

        has_word_total = has_word_positive + has_word_negative
        has_not_word_total = has_not_word_positive + has_not_word_negative

        total = has_word_total + has_not_word_total

        if has_word_total ==0 or has_not_word_total ==0:
            continue

        gini_has_word = 1 - ((has_word_positive / has_word_total) ** 2 + (has_word_negative / has_word_total) ** 2)

        gini_has_not_word = 1 - ((has_not_word_positive / has_not_word_total) ** 2 +
                                 (has_not_word_negative / has_not_word_total) ** 2)

        gini = (has_word_total / total) * gini_has_word + (has_not_word_total / total) * gini_has_not_word

        if gini_index > gini:
            gini_index = gini
            index = feature

    return gini_index, index