from threading import ExceptHookArgs
import kagglehub
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")

print("Path to dataset files:", path)

df = pd.read_csv(f"{path}/spam.csv", encoding='latin-1')[['v1', 'v2']]
df.column = ['label', 'text']
df.columns = ['label', 'text']

# Simple normalization
df['text'] = df['text'].str.lower()
df.head()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

# Convert text to numeric features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Create and train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

print("Training Success!!!")
print("Storing model ...")

try:
    joblib.dump(model, 'model/model.joblib')
    print("Successfully saved!")
    print("Saving vectorizer")
    joblib.dump(X_train_vec, "model/vectorizer.joblib")
    joblib.dump(vectorizer, "model/vectorizer.joblib")
    print("Vectorizer Saved")
except Exception as e:
    print(f"error saving model: {e}")


print("Enter Text: ")
text = input()

model = joblib.load('model/model.joblib')
vectorizer = joblib.load('model/vectorizer.joblib')

X_input = vectorizer.transform([text])

y_pred = model.predict(X_input)
print(y_pred)
