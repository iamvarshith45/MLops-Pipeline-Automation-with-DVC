import numpy as np 
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer

# Load the preprocessed train and test data
train_data = pd.read_csv('/Users/varshithmohangadupu/Desktop/MLops_DVC/data/processed_data/train_processed.csv')
test_data = pd.read_csv('/Users/varshithmohangadupu/Desktop/MLops_DVC/data/processed_data/test_processed.csv')

# Replace NaNs and strip whitespace
train_data['content'] = train_data['content'].fillna(' ').astype(str).str.strip()
test_data['content'] = test_data['content'].fillna(' ').astype(str).str.strip()

# Remove rows with empty content
train_data = train_data[train_data['content'] != '']
test_data = test_data[test_data['content'] != '']

# Check if the column names are correct
assert 'content' in train_data.columns, "Column 'content' not found in train_data"
assert 'sentiment' in train_data.columns, "Column 'sentiment' not found in train_data"

# Prepare features and labels
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(
    max_features=50,
    stop_words=None,  # features folder data use stop_words=None and features_1 folder data remove this line.
    min_df=1,                 # Include rare terms
    token_pattern=r"(?u)\b\w+\b"  # Include single-letter words if needed
)

try:
    X_train_bow = vectorizer.fit_transform(X_train)
except ValueError as e:
    print("Vectorization failed:", e)
    print("Sample training documents:", X_train[:5])
    exit(1)

X_test_bow = vectorizer.transform(X_test)

# Convert to DataFrames
train_df = pd.DataFrame(X_train_bow.toarray(), columns=vectorizer.get_feature_names_out())
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray(), columns=vectorizer.get_feature_names_out())
test_df['label'] = y_test

# Create directory to store features
data_path = os.path.join("data", "features")
os.makedirs(data_path, exist_ok=True)

# Save as CSV
train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)

print("Feature extraction using Bag of Words completed successfully.")
