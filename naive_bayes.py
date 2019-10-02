from pre_processing import get_data_for_model
from utils import cross_validation
import numpy as np
import math

#https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
class NaiveBayes:
    "Naive Bayes algorithm"
    probabilities = {}
    prior = {}

    def words_in_class(self, training, label):
        words_in_class ={}
        for i in range(len(training)):
            clazz= label[i]
            val = words_in_class.get(clazz,0)
            words_in_class[clazz]= np.add(val, training[i])
        return words_in_class

    def fit(self, training, label):
        number_of_documents = len(training)
        words_in_classes = self.words_in_class(training, label)
        classes = set(label)
        for c in classes:
            self.probabilities[c] = {}
            number_of_c = label.count(c)
            self.prior[c] = number_of_c / number_of_documents
            words_sum = sum(words_in_classes[c])
            min =0.1
            for i in range(len(training[0])):
                word = words_in_classes[c][i]
                self.probabilities[c][i] = (word +1) /(words_sum + len(training[0]))
                if self.probabilities[c][i] < min:
                    min = self.probabilities[c][i]

    def predict(self, training):
        predictions = []
        for case in training:
            scores ={}
            for clazz in self.prior.keys():
                score = math.log(self.prior[clazz])
                for word in range(len(case)):
                    word_times = case[word]
                    if word_times >0:
                        for i in range(word_times):
                            score = score + math.log(self.probabilities[clazz][word])
                scores[clazz] = score
            max_score = None
            for i in scores.keys():
                if max_score is None or scores[i]>scores[max_score]:
                    max_score = i
            predictions.append(max_score)
        return predictions


X, y = get_data_for_model()
test = NaiveBayes()
cross_validation(X,y, NaiveBayes())