import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv('health_data.csv')

# Vectorize text
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(data['Message']).toarray()

# Labels
y = data['Label']

# Save vectorizer
import pickle
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
