import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.corpus import stopwords
import re

# Read the data directly from the CSV provided
data = pd.read_csv('data_file.csv')

# Encode labels: spam as 1 and not spam as 0
data['label'] = data['label'].map({'spam': 1, 'not spam': 0})

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

# Apply cleaning function and display cleaned text
data['cleaned_text'] = data['message'].apply(clean_text)
print(data[['message', 'cleaned_text', 'label']].head())

# Vectorize the cleaned text data
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

# Evaluate Naive Bayes
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_predictions))
print(classification_report(y_test, nb_predictions))

# Test predictions and probabilities
print("Predicted probabilities for Naive Bayes:", nb_model.predict_proba(X_test))

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Evaluate Logistic Regression
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
print(classification_report(y_test, lr_predictions))

# Train a Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluate Random Forest model
print("Random Forest Model Performance:")
print(classification_report(y_test, rf_predictions))
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(y_test, rf_predictions))

# Function to predict spam
def predict_spam(message):
    cleaned_message = clean_text(message)
    vectorized_message = vectorizer.transform([cleaned_message])
    prediction = nb_model.predict(vectorized_message)
    probability = nb_model.predict_proba(vectorized_message)
    return ("Spam" if prediction[0] == 1 else "Not Spam", probability[0][1])

# Test with a sample message
print(predict_spam("Don't forget about your dentist appointment next Tuesday. Limited-time offer! Buy one and get one free on all items. Shop Now!"))
print(predict_spam("Don't forget about your dentist appointment next Tuesday. Exclusive offer! You have been selected to win a $500 gift card. Act now!"))

# Check the distribution of labels
print(data['label'].value_counts())


import joblib
# Save the model and vectorizer

spam_detector = {
    'nb_model': nb_model,
    'lr_model': lr_model,
    'rf_model': rf_model,
    'vectorizer': vectorizer,
    'clean_text': clean_text   
}

# Save the model and vectorizer to a file
joblib.dump(spam_detector, 'spam_detector.pkl')
