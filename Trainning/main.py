from preprocessing import *
from feature_extraction import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib


#Download dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
df = dataset(url)

#inspect datset
df = data_inspection(df)

#preprocess dataset
df = preprocessor(df)

#feature extraction [ Update to stop feature transformation ]
feature , label, vectorizer = feature_extraction(df)

# Build pipeline with preprocessing and model
pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', MultinomialNB())
])

# Define hyperparameter grid for GridSearchCV
param_grid = {
    'classifier__alpha': [0.01, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0],  # Smoothing parameter for Naive Bayes
}

#Perform grid search with 5-fold cross-validation and f1-score as the scoring metric
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1'
)

# Fit grid search to the data
print("Fitting the model with grid search...")
grid_search.fit(df['message'],label)

# Get the best model and its parameters
best_model = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)
print("Best f1 score: ", grid_search.best_score_)


# Save the best model to a file
model_filename = 'spam_classifier.joblib'
joblib.dump(best_model, model_filename)
print(f"Model saved to {model_filename}")