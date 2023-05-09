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
# fine tuning
svc = LinearSVC(C=1, loss="hinge")
svc.fit(x_train, y_train)


# same thing above for mutlinomialNB
mnb = MultinomialNB()
mnb.fit(x_train, y_train)

# same thing above for logistic regression
lr = LogisticRegression()
lr.fit(x_train, y_train)

# fine tuning for mnb

# ui with gradio
import gradio as gr


# def predict_review(text):
#     text = data_preprocessing(text)
#     text = vect.transform([text])
#     prediction = svc.predict(text)
#     if prediction == 1:
#         return "Positive"
#     else:
#         return "Negative"


# def svm_demo(text):
#     text = data_preprocessing(text)
#     text = vect.transform([text])
#     prediction = svc.predict(text)
#     if prediction == 1:
#         return "Positive"
#     else:
#         return "Negative"


# def mnb_demo(text):
#     text = data_preprocessing(text)
#     text = vect.transform([text])
#     prediction = mnb.predict(text)
#     if prediction == 1:
#         return "Positive"
#     else:
#         return "Negative"


# def lr_demo(text):
#     text = data_preprocessing(text)
#     text = vect.transform([text])
#     prediction = lr.predict(text)
#     if prediction == 1:
#         return "Positive"
#     else:
#         return "Negative"


# svm_demo_if = gr.Interface(
#     fn=svm_demo,
#     inputs=gr.inputs.Textbox(lines=5, placeholder="Review here..."),
#     outputs="text",
#     title="Sentiment Analysis",
#     description="Predict if a review is positive or negative",
#     theme="huggingface",
# )
# mnb_demo_if = gr.Interface(
#     fn=mnb_demo,
#     inputs=gr.inputs.Textbox(lines=5, placeholder="Review here..."),
#     outputs="text",
#     title="Sentiment Analysis",
#     description="Predict if a review is positive or negative",
#     theme="huggingface",
# )
# lr_demo_if = gr.Interface(
#     fn=lr_demo,
#     inputs=gr.inputs.Textbox(lines=5, placeholder="Review here..."),
#     outputs="text",
#     title="Sentiment Analysis",
#     description="Predict if a review is positive or negative",
#     theme="huggingface",
# )


def svc_bt(text):
    text = data_preprocessing(text)
    text = vect.transform([text])
    prediction = svc.predict(text)
    if prediction == 1:
        return "Positive"
    else:
        return "Negative"


def mnb_bt(text):
    text = data_preprocessing(text)
    text = vect.transform([text])
    prediction = mnb.predict(text)
    if prediction == 1:
        return "Positive"
    else:
        return "Negative"


def lr_bt(text):
    text = data_preprocessing(text)
    text = vect.transform([text])
    prediction = lr.predict(text)
    if prediction == 1:
        return "Positive"
    else:
        return "Negative"


# tabbed interface for multiple models
with gr.Blocks() as demo:
    gr.Markdown("Comment and uncomment the lines below to try different models")
    with gr.Tab("SVM"):
        text = gr.Textbox(lines=5, placeholder="Review here...")
        btn = gr.Button("Predict")
        label = gr.Label()
        btn.click(svc_bt, text, label)
    with gr.Tab("MultinomialNB"):
        text = gr.Textbox(lines=5, placeholder="Review here...")
        btn = gr.Button("Predict")
        label = gr.Label()
        btn.click(mnb_bt, text, label)

    with gr.Tab("Logistic Regression"):
        text = gr.Textbox(lines=5, placeholder="Review here...")
        btn = gr.Button("Predict")
        label = gr.Label()
        btn.click(lr_bt, text, label)
demo.launch()
