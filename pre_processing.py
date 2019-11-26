import hashlib

from utils import *


def diff_words(dataset, dict=None):
    if not dict:
        dict = {}
    for file in dataset:
        for word in dataset[file].split():
            word = word_filter(word)
            if not word.__eq__(''):
                dict[word] = 1
    return dict


def n_grams(dataset, clazz, BOW_SIZE, n):
    aux = []
    bag_of_grams = []
    label = []
    print("Vocabulary size: " + str(BOW_SIZE))
    for file in dataset:
        grams = [0] * BOW_SIZE
        for word in dataset[file].split():
            word = word_filter(word)
            if not word.__eq__(''):
                if n > 1:
                    aux.append(word)
                    if len(aux) >= n:
                        word = ' '.join(aux)
                        aux.pop(0)
                # https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program
                key = int(hashlib.sha512(word.encode('utf-8')).hexdigest()[:16], 16) % BOW_SIZE
                grams[key] += 1
        label.append(clazz)
        bag_of_grams.append(grams)
    return bag_of_grams, label


def get_data_for_model(bow_size, n):
    BOW_SIZE = bow_size
    if (bow_size <= 1):
        true_diff_words = diff_words(norm_true_info)
        vocabulary_size = len(diff_words(norm_false_info, true_diff_words).keys())
        print(vocabulary_size)
        BOW_SIZE = int(vocabulary_size * bow_size)
    true_grams, true_label = n_grams(norm_true_info, TRUE_NEWS, BOW_SIZE, n)
    fake_grams, fake_label = n_grams(norm_false_info, FAKE_NEWS, BOW_SIZE, n)
    return true_grams + fake_grams, true_label + fake_label
