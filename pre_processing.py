from utils import *
BOW_SIZE = 15000

def diff_words(dataset,dict=None):
    if not dict:
        dict ={}
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
                key = hash(word) % BOW_SIZE
                bow[key] += 1
        bag_of_words.append(bow)
    return bag_of_words


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


def get_data_for_model():
    bag_of_word = bag_of_words(norm_true_info) + bag_of_words(norm_false_info)
    label = [0] * len(norm_true_info) + [1] * len(norm_false_info)
    return bag_of_word,label