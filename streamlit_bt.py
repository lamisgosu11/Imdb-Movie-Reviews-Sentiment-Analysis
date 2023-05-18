import streamlit as st
import pandas as pd
import numpy as np
import pickle
from preprocessing import data_preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from preprocessing import data_preprocessing

MODEL_PATH = "D:/Coding/jptNB/HocMayThongKe/DoAn/models/"
TABS_NAME = ["📁 Type and Clean your text", "SVM", "MNB", "LR", "Voting", "XGB"]

with open("D:/Coding/jptNB/HocMayThongKe/DoAn/models/vect.pkl", "rb") as f:
    vect = pickle.load(f)
st.title("IMDB Movie Reviews Sentiment Analysis")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(TABS_NAME)
with tab1:
    text = st.text_area("Type your text here")
    if st.button("Clean"):
        text = data_preprocessing(text)
        st.write(text)
with tab2:
    with open(MODEL_PATH + "svc.pkl", "rb") as f:
        svc = pickle.load(f)
    if st.button("Predict"):
        if text:
            st.write("Your text: ", text)
            text = data_preprocessing(text)
            text = vect.transform([text])
            prediction = svc.predict(text)
            if prediction == 1:
                st.write("Positive")
            else:
                st.write("Negative")
        else:
            st.write("Please type your text first")
