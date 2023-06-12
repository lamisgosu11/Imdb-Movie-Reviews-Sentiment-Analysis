from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd 
import pickle

df = pd.read_csv("D:/Coding/jptNB/HocMayThongKe/DoAn/archive/IMDB_preprocessing.csv")

X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LinearSVC())
])

parameters = {
    "tfidf__max_df": [0.5, 0.75, 1],
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "clf__C": [1],
    "clf__loss": ['hinge'],
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)
print("Best estimator: ", grid_search.best_estimator_)
print("Best test score: ", grid_search.score(X_test, y_test))

vect = grid_search.best_estimator_["tfidf"]
clf = grid_search.best_estimator_["clf"]

with open("D:/Coding/jptNB/HocMayThongKe/DoAn/hyperparameters/vect_linearSVC.pkl", "wb") as f:
    pickle.dump(vect, f)
with open("D:/Coding/jptNB/HocMayThongKe/DoAn/hyperparameters/clf_LinearSVC.pkl", "wb") as f:
    pickle.dump(clf, f)