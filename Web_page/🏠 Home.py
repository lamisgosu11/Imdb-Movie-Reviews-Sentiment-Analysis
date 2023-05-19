import streamlit as st

st.set_page_config(
    page_title="Streamlit App",
    page_icon="🏠",
)

st.write("# IMDB Movie Reviews Sentiment Analysis 🎬")

st.sidebar.success("Select a Demo to start.")

st.markdown(
    """
    - Sentiment analysis of movie reviews is a vital task in understanding the overall reception of films among audiences. 
    
    - In this project, we employ various machine learning models, including Support Vector Machines (SVM), Multinomial Naive Bayes, Logistic Regression, and a Voting Classifier, to classify movie reviews based on their sentiment. 
    - The objective is to build accurate models that can automatically predict whether a given review is positive or negative.

    **👈 Select a demo from the sidebar** to see some examples
    ### About our dataset
    - Check out [Kaggle 💾](https://shorturl.at/aiqxS)

    - IMDB dataset having 50K movie reviews for natural language processing or Text analytics.
    - This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. 
    - We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. 
    - So, predict the number of positive and negative reviews using either classification or deep learning algorithms.

"""
)
