from sklearn.ensemble import RandomForestClassifier
from pre_processing import get_data_for_model
from utils import cross_validation

#based on: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
#tunning1: https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
#tunnning2: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
#pesquisar técnicas de tunning
X, y = get_data_for_model()
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
cross_validation(X,y, rf)