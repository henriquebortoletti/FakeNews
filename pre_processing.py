from utils import *
BOW_SIZE = 30000


def diff_words(dataset,dict=None):
    if not dict:
        dict ={}
    for file in dataset:
        for word in dataset[file].split():
            word = word_filter(word)
            if not word.__eq__(''):
                dict[word] = 1
    return dict


def bag_of_words(dataset,bow = None):
    if not bow:
        bow = [0] * BOW_SIZE
    for file in dataset:
        for word in dataset[file].split():
            word = word_filter(word)
            if not word.__eq__(''):
                key = hash(word) % BOW_SIZE
                bow[key] += 1
    return bow


def n_grams(dataset,grams = None, n= 2):
    if not grams:
        grams = [0] * BOW_SIZE
    aux = []
    for file in dataset:
        for word in dataset[file].split():
            word = word_filter(word)
            if not word.__eq__(''):
                aux.append(word)
                if len(aux) >= n:
                    key = ' '.join(aux)
                    print(key)
                    key = hash(key) % BOW_SIZE
                    grams[key] += 1
                    aux.pop(0)
    return grams
