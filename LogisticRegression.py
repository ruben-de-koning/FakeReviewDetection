import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from Kaggle: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset?resource=download
data = pd.read_csv('data/reviews.csv')

# Convert labels to 0 and 1
data['label'] = data['label'].map({'OR': 0, 'CG': 1})

# Download nltk
nltk.download('punkt')
nltk.download('stopwords')

# Define a function to clean the text
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Clean the text of the review
data['cleaned_text'] = data['text'].apply(clean_text)
data_cleaned = data.dropna(subset=['label'])

# Now split the cleaned data, not the original data
X_train, X_test, y_train, y_test = train_test_split(
    data_cleaned['cleaned_text'], data_cleaned['label'], test_size=0.15, random_state=42, stratify=data_cleaned['label'], shuffle=True
)

# Initialize and use the TF-IDF vectorizer with a limit of 5k features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Make prediction on the test part
y_pred = model.predict(X_test_tfidf)

# Compute and plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
