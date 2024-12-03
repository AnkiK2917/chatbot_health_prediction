import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# Load the health dataset (ensure it's structured correctly with 'Message' and 'Label' columns)
df = pd.read_csv('health_data.csv')

# Preprocessing: Remove NaN values from the 'Label' column
df = df.dropna(subset=['Label'])

# Initialize the vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Transform the text data (messages) into numeric features
X = vectorizer.fit_transform(df['Message']).toarray()

# Map health condition labels to numeric values
label_mapping = {
    'anxiety': 0, 'depression': 1, 'stress': 2, 'pain': 3, 'fatigue': 4,
    'physical stress': 5, 'nutrition': 6, 'neutral': 7
}
y = df['Label'].map(label_mapping).values

# Split data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Decision Tree Classifier
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)  # Set max_depth for better generalization
dt_model.fit(X_train, y_train)

# Predict and evaluate Decision Tree
y_dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_dt_pred)
print(f"Decision Tree Accuracy: {dt_accuracy * 100:.2f}%")
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_dt_pred))

# Save the Decision Tree model
with open('health_model.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

### Support Vector Machine (SVM) Classifier
svm_model = SVC(kernel='linear', probability=True, random_state=42)  # Linear kernel for text classification
svm_model.fit(X_train, y_train)

# Predict and evaluate SVM
y_svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_svm_pred)
print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")
print("SVM Classification Report:")
print(classification_report(y_test, y_svm_pred))

# Save the SVM model
with open('health_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)

# Save the vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("SVM and Decision Tree models trained and saved!")
