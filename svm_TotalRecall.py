# Step 1: Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, recall_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np

# Step 2: Load your dataset
# Assuming `data` is your DataFrame containing the text data and labels
# `text_column` is the name of the column with the text
# `label_column` is the name of the column with the labels indicating the presence of psychological disorders
texts = data[text_column]
labels = data[label_column]

# Step 3: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Step 4: Define a pipeline for text preprocessing and classification
# The TfidfVectorizer converts text to a matrix of TF-IDF features.
# The SVC model is adjusted to optimize for high recall.
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('svc', SVC(class_weight='balanced', probability=True)) # Using 'balanced' to adjust weights inversely proportional to class frequencies
])

# Step 5: Define a scoring function that focuses on recall
recall_optimization = make_scorer(recall_score, greater_is_better=True)

# Step 6: Use GridSearchCV to find the best parameters for maximizing recall
param_grid = {
    'svc__C': [0.1, 1, 10], # Regularization parameter
    'svc__kernel': ['linear', 'rbf'], # Kernel type
    'svc__gamma': ['scale', 'auto'], # Kernel coefficient
}

# The GridSearchCV instance
grid_search = GridSearchCV(pipeline, param_grid, scoring=recall_optimization, cv=5)

# Step 7: Fit the model
grid_search.fit(X_train, y_train)

# Step 8: Evaluate the model
predictions = grid_search.predict(X_test)
print(classification_report(y_test, predictions))

# The recall score is specifically printed to ensure it meets the objective
print("Recall Score:", recall_score(y_test, predictions))

# Note: The primary goal is to achieve a recall of 1.00. The parameters in the grid (C, kernel, gamma) are adjusted to explore a range of possibilities. 
# The 'class_weight' parameter in SVC is set to 'balanced' to automatically adjust weights inversely proportional to class frequencies, which is crucial for handling imbalanced datasets often found in psychological disorder detection tasks.
# This setup might require further refinement based on the initial results, including adjusting the parameter grid, considering different feature extraction methods, or employing more advanced techniques like SMOTE for handling imbalanced data.
