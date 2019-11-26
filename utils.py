import re
from string import punctuation
from nltk.corpus import stopwords
from dataset_reader import *

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
    #a regex é feita para palavras que possuem a pontuação colada por exemplo henrique,
    word = re.sub(r'[^\w\s]', '', word).lower()
    if word not in stop_words and word.isalpha():
        return word
    else:
        return ''




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

