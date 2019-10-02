from utils import *
import os
import hashlib
BOW_SIZE = 15000


def diff_words(dataset, dict=None):
    if not dict:
        dict = {}
    for file in dataset:
        for word in dataset[file].split():
            word = word_filter(word)
            if not word.__eq__(''):
                dict[word] = 1
    return dict


def bag_of_words(dataset):
    bag_of_words = []
    for file in dataset:
        bow = [0] * BOW_SIZE
        for word in dataset[file].split():
            word = word_filter(word)
            if not word.__eq__(''):
                #https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program
                key = int(hashlib.sha512(word.encode('utf-8')).hexdigest()[:16], 16)%BOW_SIZE
                bow[key] += 1
        bag_of_words.append(bow)
    return bag_of_words


bag_of_word = bag_of_words(norm_true_info) + bag_of_words(norm_false_info)
label = [TRUE_NEWS] * len(norm_true_info) + [FAKE_NEWS] * len(norm_false_info)


def n_grams(dataset,clazz ,n=2):
    aux = []
    bag_of_grams = []
    label =[]
    for file in dataset:
        grams = [0] * BOW_SIZE
        for word in dataset[file].split():
            word = word_filter(word)
            if not word.__eq__(''):
                aux.append(word)
                if len(aux) >= n:
                    key = ' '.join(aux)
                    key = hash(key) % BOW_SIZE
                    grams[key] += 1
                    aux.pop(0)
        label.append(clazz)
        bag_of_grams.append(grams)
    return bag_of_grams,label


def get_data_for_model(n=0):
    if n >=2:
        true_grams, true_label = n_grams(norm_true_info,TRUE_NEWS,n)
        fake_grams, fake_label = n_grams(norm_false_info,FAKE_NEWS,n)
        return true_grams+fake_grams,true_label+fake_label
    return bag_of_word, label

