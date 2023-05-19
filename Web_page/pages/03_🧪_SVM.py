from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from preprocessing import data_preprocessing
import pickle
import streamlit as st

st.set_page_config(page_title="Support Vector Machine", page_icon="🧪")
with open("D:/Coding/jptNB/HocMayThongKe/DoAn/models/vect.pkl", "rb") as f:
    vect = pickle.load(f)
with open("D:/Coding/jptNB/HocMayThongKe/DoAn/models/svc.pkl", "rb") as f:
    svm = pickle.load(f)
st.header("Multinomial Naive Bayes")
text = st.session_state.text
clean_text = st.session_state.clean_text
st.write("- This is your text after Preprocessing:")
st.write(clean_text)
text = vect.transform([clean_text])
if st.button("Classify"):
    prediction = svm.predict(text)
    if prediction == 1:
        st.write("Positive")
    else:
        st.write("Negative")
