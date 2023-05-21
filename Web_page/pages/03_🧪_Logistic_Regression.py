from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from preprocessing import data_preprocessing
import pickle
import streamlit as st

st.set_page_config(
    page_title="Logistic Regression",
    page_icon="🧪",
    layout="wide",
)
with open("D:/Coding/jptNB/HocMayThongKe/DoAn/models/vect.pkl", "rb") as f:
    vect = pickle.load(f)
with open("D:/Coding/jptNB/HocMayThongKe/DoAn/models/logreg.pkl", "rb") as f:
    logreg = pickle.load(f)
st.header("Multinomial Naive Bayes")
text = st.session_state.text
clean_text = st.session_state.clean_text
st.write("- This is your text after Preprocessing:")
st.write(clean_text)
text = vect.transform([clean_text])
if st.button("Classify"):
    prediction = logreg.predict(text)
    if prediction == 1:
        st.write("Positive")
    else:
        st.write("Negative")
# adding button for probability prediction
if st.button("Probability"):
    probability = logreg.predict_proba(text)
    st.write("This is the probability of your text being positive or negative: ")
    # adding label for probability
    st.write("Positive: ", probability[0][0])
    st.write("Negative: ", probability[0][1])

