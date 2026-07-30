import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags (#)
    text = re.sub(r"#", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = " ".join(text.split())

    # Tokenize
    words = text.split()

    # Remove stopwords and apply lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)



# Get the path of the current file
current_dir = os.path.dirname(__file__)

# Go to the project root
project_dir = os.path.dirname(current_dir)

# Dataset folder path
dataset_path = os.path.join(project_dir, "dataset")

# Read the dataset files
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
# Remove duplicate rows
train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)
val_df.drop_duplicates(inplace=True)

print("Duplicates removed successfully!")


# Apply preprocessing
train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)
val_df["clean_text"] = val_df["text"].apply(clean_text)

print("\nOriginal Text:")
print(train_df["text"].iloc[0])

print("\nCleaned Text:")
print(train_df["clean_text"].iloc[0])

print("Train Dataset Shape:", train_df.shape)
print("Test Dataset Shape:", test_df.shape)
print("Validation Dataset Shape:", val_df.shape)

print("\nFirst 5 Rows")
print(train_df.head())

print("\nDataset Information")
train_df.info()

print("\nMissing Values")
print(train_df.isnull().sum())

print("\nDuplicate Rows")
print(train_df.duplicated().sum())

print("\nEmotion Counts")
print(train_df["emotion"].value_counts())

# Emotion Distribution
plt.figure(figsize=(8,5))

sns.countplot(
    x="emotion",
    data=train_df,
    order=train_df["emotion"].value_counts().index
)

plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")

plt.show()