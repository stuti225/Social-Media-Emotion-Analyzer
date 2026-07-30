import streamlit as st
import pandas as pd
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------
# Download NLTK Resources
# ---------------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# ---------------------------------
# Initialize NLP Objects
# ---------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# ---------------------------------
# Load Trained Model & Vectorizer
# ---------------------------------
model = joblib.load("models/emotion_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# ---------------------------------
# Emotion Emojis
# ---------------------------------
emotion_emojis = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲"
}

# ---------------------------------
# Text Cleaning Function
# ---------------------------------
def clean_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags
    text = re.sub(r"#", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = " ".join(text.split())

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords and lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ---------------------------------
# Streamlit Configuration
# ---------------------------------
st.set_page_config(
    page_title="Social Media Emotion Analyzer",
    page_icon="😊",
    layout="centered"
)

# ---------------------------------
# Title
# ---------------------------------
st.title("😊 Social Media Emotion Analyzer")

st.write(
    "Enter a social media post below and predict its emotion using a Machine Learning model."
)

# ---------------------------------
# Initialize Prediction History
# ---------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------
# User Input
# ---------------------------------
user_input = st.text_area(
    "Enter Text",
    height=150,
    placeholder="Example: I am feeling very excited today!"
)

# ---------------------------------
# Prediction
# ---------------------------------
if st.button("Predict Emotion"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        cleaned_text = clean_text(user_input)

        vector = vectorizer.transform([cleaned_text])

        prediction = model.predict(vector)[0]

        emoji = emotion_emojis.get(prediction, "🙂")

        st.success(
            f"{emoji} Predicted Emotion: **{prediction.capitalize()}**"
        )
        st.subheader("Processed Text")

        st.code(cleaned_text)
        
        # Save prediction history
        st.session_state.history.append({
            "Text": user_input,
            "Predicted Emotion": f"{emoji} {prediction.capitalize()}"
        })

# ---------------------------------
# Prediction History
# ---------------------------------
if st.session_state.history:

    st.subheader("Prediction History")

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df, use_container_width=True)

# ---------------------------------
# Clear History Button
# ---------------------------------
if st.session_state.history:

    if st.button("Clear History"):

        st.session_state.history = []

        st.rerun()