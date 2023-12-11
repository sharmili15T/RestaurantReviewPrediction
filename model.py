import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('restaurant_reviews.tsv', delimiter='\t', encoding='utf-8')
# Preprocess the text (you may need to do further text cleaning)
data['review'] = data['review'].apply(lambda x: x.lower())
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust the max_features as needed
# Fit and transform the training data
X_train_vectors = tfidf_vectorizer.fit_transform(X_train)
# Transform the testing data
X_test_vectors = tfidf_vectorizer.transform(X_test)

#Model building
from sklearn.linear_model import LogisticRegression
# Create and train the model
model = LogisticRegression()
model.fit(X_train_vectors, y_train)

#Model evaluation
from sklearn.metrics import accuracy_score, classification_report
# Predict on the test set
y_pred = model.predict(X_test_vectors)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
