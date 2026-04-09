import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================
# LOAD DATASET (Working URL)
# =========================
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/imdb.txt"
df = pd.read_csv(url, sep="\t", header=None)
df.columns = ["review", "sentiment"]

# =========================
# EDA
# =========================
print(df.head(10))
print("Shape:", df.shape)
print(df.info())
print(df.describe())

print("Missing:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())

print("Count:\n", df["sentiment"].value_counts())

plt.figure()
sns.countplot(x=df["sentiment"])
plt.title("Positive vs Negative Reviews")
plt.show()

# =========================
# TEXT PREPROCESSING (TF-IDF)
# =========================
X = df["review"]
y = df["sentiment"]

tfidf = TfidfVectorizer(stop_words="english")
X_tfidf = tfidf.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# =========================
# DECISION TREE MODEL
# =========================
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

# =========================
# EVALUATION
# =========================
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# DECISION TREE VISUALIZATION
# =========================
plt.figure(figsize=(15,8))
plot_tree(dt, max_depth=3, filled=True)
plt.show()

# =========================
# OVERFITTING CHECK
# =========================
print("Train Accuracy:", dt.score(X_train, y_train))
print("Test Accuracy:", dt.score(X_test, y_test))

# =========================
# CONCLUSION
# =========================
print("\nConclusion:")
print("Decision Tree is used to classify movie reviews into positive or negative.")
print("TF-IDF converts text into numerical features.")
print("Model performance evaluated using standard metrics.")
