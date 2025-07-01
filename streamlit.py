import streamlit as st
import kagglehub
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("📩 SMS Spam Detector with Naive Bayes")

# Download dataset only once


@st.cache_data
def load_data():
    try:
        model = joblib.load('model/model.joblib')
        vectorizer = joblib.load('model/vectorizer.joblib')
    except Exception as e:
        print(f"Error in loading model :( {e}")
        return None, None

    return model, vectorizer

# Train model and save


# @st.cache_resource
# def train_and_save_model(df):
#     X_train, X_test, y_train, y_test = train_test_split(
#         df['text'], df['label'], test_size=0.2, random_state=42
#     )
#
#     vectorizer = CountVectorizer()
#     X_train_vec = vectorizer.fit_transform(X_train)
#
#     model = MultinomialNB()
#     model.fit(X_train_vec, y_train)
#
#     os.makedirs("model", exist_ok=True)
#     joblib.dump(model, "model/model.joblib")
#     joblib.dump(vectorizer, "model/vectorizer.joblib")
#
#     return model, vectorizer


# Load data and train model
with st.spinner("Loading model..."):
    model, vectorizer = load_data()

if not model and not vectorizer:
    st.error("Model loading failed")

st.success("Model is ready!")

# Text input
user_input = st.text_input("📨 Enter an SMS message to classify:")

if user_input and model and vectorizer:
    X_input = vectorizer.transform([user_input.lower()])
    prediction = model.predict(X_input)[0]
    st.write(f"🧠 Prediction: **{prediction.upper()}**")
