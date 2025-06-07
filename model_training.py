
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

# Load the dataset (update the path accordingly)
df = pd.read_csv('/content/Resume.csv')

# Retain only necessary columns and drop rows with missing values
df = df[['ID', 'Resume_str', 'Resume_html', 'Category']].dropna()

# Bar plot for category distribution
plt.figure(figsize=(15, 5))
sns.countplot(x='Category', data=df)
plt.xticks(rotation=90)
plt.title("Resume Category Distribution")
plt.savefig('static/images/category_distribution.png')
plt.close()

# Pie chart
category_counts = df['Category'].value_counts()
plt.figure(figsize=(10, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%',
        shadow=False, colors=plt.cm.plasma(np.linspace(0, 1, len(category_counts))))
plt.title("Resume Category Share")
plt.savefig('static/images/category_piechart.png')
plt.close()

max_count = df['Category'].value_counts().max()
balanced_data = []

for category in df['Category'].unique():
    category_data = df[df['Category'] == category]
    resampled_data = resample(category_data, replace=True, n_samples=max_count, random_state=42)
    balanced_data.append(resampled_data)

balanced_df = pd.concat(balanced_data)
balanced_df.dropna(inplace=True)

X = balanced_df['Resume_str']           # Resume content
y = balanced_df['Category']             # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

y_pred = rf_classifier.predict(X_test_tfidf)
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
            xticklabels=rf_classifier.classes_,
            yticklabels=rf_classifier.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('static/images/confusion_matrix.png')
plt.close()

pickle.dump(rf_classifier, open('models/rf_classifier.pkl', 'wb'))
pickle.dump(tfidf_vectorizer, open('models/tfidf_vectorizer.pkl', 'wb'))

import os
import pickle

# Updated full path
base_path = r"C:\Users\This PC\OneDrive\OneDrive - Islamabad Model Postgraduate College of Commerce H-8 4 Islamabad\SZABIST WORKING ZONE\Project\myproject\models"

# Make sure the path exists
os.makedirs(base_path, exist_ok=True)

# Define full file paths
model_path = os.path.join(base_path, 'rf_classifier.pkl')
vectorizer_path = os.path.join(base_path, 'tfidf_vectorizer.pkl')

# Save the model and vectorizer
pickle.dump(rf_classifier, open(model_path, 'wb'))
pickle.dump(tfidf_vectorizer, open(vectorizer_path, 'wb'))

print("Model saved at:", model_path)
print("Vectorizer saved at:", vectorizer_path)

