from sklearn.ensemble import RandomForestClassifier

from validation import extract_metrics

#based on: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

extract_metrics(rf,"random_forest")