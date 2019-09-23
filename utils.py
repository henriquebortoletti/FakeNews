from dataset_reader import *
from nltk.corpus import stopwords
from string import punctuation
import re

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


def word_filter(word):
    word = re.sub(r'[^\w\s]', '', word).lower()
    if word not in stop_words and word.isalpha():
        return word
    else:
        return ''
