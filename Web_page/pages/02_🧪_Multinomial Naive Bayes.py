from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from preprocessing import data_preprocessing
import pickle
import streamlit as st

st.set_page_config(page_title="Multinomial Naive Bayes", page_icon="🧪")
with open("D:/Coding/jptNB/HocMayThongKe/DoAn/models/vect.pkl", "rb") as f:
    vect = pickle.load(f)
with open("D:/Coding/jptNB/HocMayThongKe/DoAn/models/mnb.pkl", "rb") as f:
    mnb = pickle.load(f)
st.header("Multinomial Naive Bayes")
text = st.session_state.text
clean_text = st.session_state.clean_text
st.write("- This is your text after Preprocessing:")
st.write(clean_text)
text = vect.transform([clean_text])
if st.button("Classify"):
    prediction = mnb.predict(text)
    if prediction == 1:
        st.write("Positive")
    else:
        st.write("Negative")
# adding button for probability prediction
if st.button("Probability"):
    probability = mnb.predict_proba(text)
    st.write("This is the probability of your text being positive or negative: ")
    # adding label for probability
    st.write("Positive: ", probability[0][0])
    st.write("Negative: ", probability[0][1])
