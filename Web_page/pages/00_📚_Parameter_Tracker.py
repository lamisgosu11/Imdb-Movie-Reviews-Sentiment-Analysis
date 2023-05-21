import pickle
import streamlit as st

st.set_page_config(
    page_title="Parameter Tracker",
    page_icon="📚",
    layout="wide",
)
if "text" not in st.session_state:
    st.session_state.text = "Default"
if "clean_text" not in st.session_state:
    st.session_state.clean_text = "Default"
text = st.text_area("Change your text here")
clean_text = st.text_area("Change your clean text here")
update = st.button("Update")
if update:
    st.session_state.text = text
    st.session_state.clean_text = clean_text
button = st.button("Currently, this is your parameters: ")
if button:
    st.write("text: ", st.session_state.text)
    st.write("clean_text: ", st.session_state.clean_text)
