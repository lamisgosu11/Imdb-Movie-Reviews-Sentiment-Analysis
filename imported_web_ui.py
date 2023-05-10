# import models with pickle
import pickle
import os
import pandas as pd
import numpy as np
import gradio as gr
from preprocessing import data_preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

# load models
df = pd.read_csv("/archive/IMDB_preprocessing.csv")
X = df["review"]
Y = df["sentiment"]
vect = TfidfVectorizer()
X = vect.fit_transform(df["review"])
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)
# import models from pickle files
with open("/models/svc.pkl", "rb") as f:
    svc = pickle.load(f)
with open("/models/mnb.pkl", "rb") as f:
    mnb = pickle.load(f)
with open("/models/logreg.pkl", "rb") as f:
    lr = pickle.load(f)
with open("/models/voting_clf.pkl", "rb") as f:
    voting_clf = pickle.load(f)
with open("/models/xgb.pkl", "rb") as f:
    xgb = pickle.load(f)


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


def voting_clf_bt(text):
    text = data_preprocessing(text)
    text = vect.transform([text])
    prediction = voting_clf.predict(text)
    if prediction == 1:
        return "Positive"
    else:
        return "Negative"


with gr.Blocks() as demo:
    gr.Markdown("Sentiment Analysis Demo")
    with gr.Tab("SVC"):
        text = gr.Textbox(lines=5, placeholder="Type your review here...")
        btn = gr.Button("Classify")
        label = gr.Label()
        btn.click(svc_bt, text, label)
        gr.Markdown("## Examples")
        gr.Examples(
            examples=[
                "I love this movie",
                "I hate this movie",
                "I don't know what to say about this movie",
            ],
            inputs=text,
            outputs=label,
            fn=svc_bt,
            cache_examples=True,
        )
    with gr.Tab("Multinomial Naive Bayes"):
        text = gr.Textbox(lines=5, placeholder="Type your review here...")
        btn = gr.Button("Classify")
        label = gr.Label()
        btn.click(mnb_bt, text, label)
        gr.Markdown("## Examples")
        gr.Examples(
            examples=[
                "I love this movie",
                "I hate this movie",
                "I don't know what to say about this movie",
            ],
            inputs=text,
            outputs=label,
            fn=mnb_bt,
            cache_examples=True,
        )

    with gr.Tab("Logistic Regression"):
        text = gr.Textbox(lines=5, placeholder="Type your review here...")
        btn = gr.Button("Classify")
        label = gr.Label()
        btn.click(lr_bt, text, label)
        gr.Markdown("## Examples")
        gr.Examples(
            examples=[
                "I love this movie",
                "I hate this movie",
                "I don't know what to say about this movie",
            ],
            inputs=text,
            outputs=label,
            fn=lr_bt,
            cache_examples=True,
        )
    with gr.Tab("Voting Classifier"):
        text = gr.Textbox(lines=5, placeholder="Type your review here...")
        btn = gr.Button("Classify")
        label = gr.Label()
        btn.click(voting_clf_bt, text, label)
        gr.Markdown("## Image Examples")
        gr.Examples(
            examples=[
                "I love this movie",
                "I hate this movie",
                "I don't know what to say about this movie",
            ],
            inputs=text,
            outputs=label,
            fn=voting_clf_bt,
            cache_examples=True,
        )


demo.launch()
