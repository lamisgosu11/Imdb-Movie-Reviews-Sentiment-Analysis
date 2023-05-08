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

warnings.filterwarnings("ignore")

stop_words = set(stopwords.words("english"))

df = pd.read_csv("/archive/IMDB Dataset.csv")


def no_of_words(text):
    words = text.split()
    word_count = len(words)
    return word_count


df["word_count"] = df["review"].apply(no_of_words)
df.sentiment.replace("positive", 1, inplace=True)
df.sentiment.replace("negative", 2, inplace=True)


# **Prepocessing**
def data_preprocessing(text):
    text = text.lower()
    text = re.sub("<br />", "", text)
    text = re.sub(r"https\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


df.review = df["review"].apply(data_preprocessing)
df = df.drop_duplicates("review")
stemmer = PorterStemmer()


def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


df.review = df["review"].apply(lambda x: stemming(x))
df["word_count"] = df["review"].apply(no_of_words)

# export to csv
df.to_csv("/archive/IMDB_preprocessing.csv", index=False)
