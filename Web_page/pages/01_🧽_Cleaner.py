import pickle
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import sys

sys.path.append("D:\Coding\jptNB\HocMayThongKe\DoAn")
from preprocessing import data_preprocessing


st.set_page_config(
    page_title="Text Cleaner",
    page_icon="🧽",
    layout="wide",
)
text = st.session_state.text

with open("D:/Coding/jptNB/HocMayThongKe/DoAn/models/vect.pkl", "rb") as f:
    vect = pickle.load(f)
st.header("Type your text here and I will clean it for you 🧹")
text = st.text_area("Type to change your text here")
st.session_state.text = text
if st.button("Clean"):
    clean_text = data_preprocessing(text)
    st.session_state.clean_text = clean_text
    st.write("This is your text after we clean it for you: \n")
    st.write(clean_text)
