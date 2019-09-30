from pre_processing import get_data_for_model
from utils import cross_validation


class DecisionTree:
    "Decision Tree algorithm"
    weights=[]
    def fit(self,features, label):
        print("fit")
        return label

    def predict(self,features):
        print("predict")
        return self.weights


X, y = get_data_for_model()
cross_validation(X,y, DecisionTree())
