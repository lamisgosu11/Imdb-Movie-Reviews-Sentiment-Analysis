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

# import preprocessing
from preprocessing import data_preprocessing

warnings.filterwarnings("ignore")

stop_words = set(stopwords.words("english"))
# import dataset
df = pd.read_csv("/archive/IMDB_preprocessing.csv")
# Factorize
X = df["review"]
Y = df["sentiment"]
vect = TfidfVectorizer()
X = vect.fit_transform(df["review"])
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

svc = LinearSVC()
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)

# fine tuning
svc = LinearSVC(C=1, loss="hinge")
svc.fit(x_train, y_train)
svc_pred = svc.predict(x_test)

# ui with gradio
import gradio as gr


def predict_review(text):
    text = data_preprocessing(text)
    text = vect.transform([text])
    prediction = svc.predict(text)
    if prediction == 1:
        return "Positive"
    else:
        return "Negative"


iface = gr.Interface(
    fn=predict_review,
    inputs=gr.inputs.Textbox(lines=5, placeholder="Review here..."),
    outputs="text",
    title="Sentiment Analysis",
    description="Predict if a review is positive or negative",
    theme="huggingface",
)


iface.launch()
