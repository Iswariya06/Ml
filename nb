import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# =========================
# LOAD DATASET
# =========================
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
df = pd.read_csv(url, header=None)

# =========================
# EDA
# =========================
print(df.head(10))
print("\nShape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nDescribe:")
print(df.describe())

print("\nMissing values:", df.isnull().sum().sum())
print("Duplicates:", df.duplicated().sum())

print("\nClass distribution:")
print(df[df.columns[-1]].value_counts())

plt.figure()
sns.countplot(x=df[df.columns[-1]])
plt.title('Spam vs Not Spam')
plt.show()

# =========================
# SPLIT DATA
# =========================
X = df.drop(df.columns[-1], axis=1)
y = df[df.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =========================
# MODEL (NAIVE BAYES)
# =========================
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

# =========================
# EVALUATION
# =========================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

plt.figure()
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# =========================
# TP, FP, FN, TN
# =========================
tn, fp, fn, tp = cm.ravel()

print('\nTrue Negative:', tn)
print('False Positive:', fp)
print('False Negative:', fn)
print('True Positive:', tp)

# =========================
# INTERPRETATION
# =========================
print("\nInterpretation:")
print("False Positive → Spam classified as Not Spam")
print("False Negative → Not Spam classified as Spam")

# =========================
# CONCLUSION
# =========================
print("\nConclusion:")
print("Naive Bayes model successfully classifies emails as Spam or Not Spam.")
