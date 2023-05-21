# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import plotly.express as px
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
import pickle
from preprocessing import data_preprocessing

warnings.filterwarnings("ignore")
# import dataset
df = pd.read_csv("archive/IMDB_preprocessing.csv")
# Factorize
X = df["review"]
Y = df["sentiment"]
vect = TfidfVectorizer()
X = vect.fit_transform(df["review"])
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
# models with finest parameters
# Linear SVC
svc = LinearSVC(C=0.1, random_state=42)
svc.fit(x_train, y_train)

# Multinomial Naive Bayes
mnb = MultinomialNB(alpha=1)
mnb.fit(x_train, y_train)

# Logistic Regression
logreg = LogisticRegression(C=10)
logreg.fit(x_train, y_train)

from sklearn.calibration import CalibratedClassifierCV

calibrated_svc = CalibratedClassifierCV(svc)
calibrated_svc.fit(x_train, y_train)
# ENSEMBLE MODELS

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ("Logistic Regression", logreg),
        ("Multinomial Naive Bayes", mnb),
        ("Linear SVC", svc),
    ],
    voting="hard",
)
voting_clf.fit(x_train, y_train)

voting_clf_soft = VotingClassifier(
    estimators=[
        ("Logistic Regression", logreg),
        ("Multinomial Naive Bayes", mnb),
        ("Linear SVC", calibrated_svc),
    ],
    voting="soft",
)
voting_clf_soft.fit(x_train, y_train)
# save model (pickle)
# vectorizer
pickle.dump(vect, open("models/vectorizer.pkl", "wb"))
pickle.dump(logreg, open("models/logreg.pkl", "wb"))
pickle.dump(mnb, open("models/mnb.pkl", "wb"))
pickle.dump(svc, open("models/svc.pkl", "wb"))
pickle.dump(voting_clf, open("models/voting_clf.pkl", "wb"))
pickle.dump(voting_clf_soft, open("models/voting_clf_soft.pkl", "wb"))
pickle.dump(calibrated_svc, open("models/calibrated_svc.pkl", "wb"))
