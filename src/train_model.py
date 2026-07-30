import os
import re
import string
import joblib
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from evaluate_model import evaluate_model

# -------------------------------
# Download required NLTK resources
# -------------------------------
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# -------------------------------
# Initialize objects
# -------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -------------------------------
# Text Cleaning Function
# -------------------------------
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

    # Remove stopwords and apply lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -------------------------------
# Load Dataset
# -------------------------------
current_dir = os.path.dirname(__file__)
project_dir = os.path.dirname(current_dir)
dataset_path = os.path.join(project_dir, "dataset")

train_df = pd.read_csv(
    os.path.join(dataset_path, "train.txt"),
    sep=";",
    names=["text", "emotion"]
)

test_df = pd.read_csv(
    os.path.join(dataset_path, "test.txt"),
    sep=";",
    names=["text", "emotion"]
)

val_df = pd.read_csv(
    os.path.join(dataset_path, "val.txt"),
    sep=";",
    names=["text", "emotion"]
)

# -------------------------------
# Remove Duplicates
# -------------------------------
train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)
val_df.drop_duplicates(inplace=True)

print("Duplicates removed successfully!")

# -------------------------------
# Text Preprocessing
# -------------------------------
train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)
val_df["clean_text"] = val_df["text"].apply(clean_text)

print("\nOriginal Text:")
print(train_df["text"].iloc[0])

print("\nCleaned Text:")
print(train_df["clean_text"].iloc[0])

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X_train = vectorizer.fit_transform(train_df["clean_text"])
X_test = vectorizer.transform(test_df["clean_text"])
X_val = vectorizer.transform(val_df["clean_text"])

y_train = train_df["emotion"]
y_test = test_df["emotion"]
y_val = val_df["emotion"]

print("\nTF-IDF Matrix Shape:", X_train.shape)

# -------------------------------
# Train Linear SVM
# -------------------------------
model = LinearSVC(random_state=42)

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# -------------------------------
# Prediction
# -------------------------------
y_pred = model.predict(X_test)

print("\nPrediction completed successfully!")

# -------------------------------
# Model Evaluation
# -------------------------------
evaluate_model(y_test, y_pred)

# -------------------------------
# Save Model
# -------------------------------
model_path = os.path.join(project_dir, "models", "emotion_model.pkl")
vectorizer_path = os.path.join(project_dir, "models", "tfidf_vectorizer.pkl")

joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)

print("\nModel saved successfully!")

