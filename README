# SMS Spam Classification

A machine learning project to classify SMS messages as spam or legitimate (ham) communications using Natural Language Processing and Naive Bayes classification.

## Project Overview

This project demonstrates how to build and train a machine learning model to identify spam SMS messages. The implementation includes data preprocessing, feature extraction, model training with hyperparameter optimization, and a testing framework to evaluate the model's performance on unseen data.

## Acknowledgment

This project is based on materials from HTB Academy's "Applications of AI in InfoSec" module. The original codebase and learning materials are provided by Hack The Box Academy, and this implementation builds upon their educational resources.

## Features

- Data preprocessing pipeline for text normalization
- Feature extraction using TF-IDF vectorization
- Naive Bayes classification with hyperparameter tuning
- Model evaluation with F1 score metric
- Testing framework for evaluating model performance on new messages

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sms-spam-classification.git
cd sms-spam-classification
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

### Dataset

The project uses the SMS Spam Collection dataset from UCI Machine Learning Repository. The dataset will be automatically downloaded and preprocessed when running the main script.

## Usage Guide

### Training the Model

To train the spam classification model:

```bash
python main.py
```

This will:
1. Download the dataset
2. Preprocess the text data
3. Extract features
4. Train a Naive Bayes classifier with grid search for hyperparameter optimization
5. Save the trained model to `spam_classifier.joblib`

### Testing the Model

To test the trained model on sample messages:

```bash
python test.py
```

By default, this will test the model on the last 15 messages from the test dataset. You can modify the `sample_size` and `sample_method` parameters in `test.py` to test different portions of the dataset:

```python
# Define test parameters
sample_size = 15  # Number of messages to test
sample_method = "tail"  # Options: "head", "tail", or set to another value for random sampling
```

### Using the Model in Your Own Applications

To use the trained model in your own applications:

```python
import joblib
import pandas as pd
from preprocessing import preprocessor

# Load the trained model
model = joblib.load('spam_classifier.joblib')

# Create a DataFrame with your message
message = "Your message here"
df = pd.DataFrame({'message': [message]})

# Preprocess your message
processed_df = preprocessor(df)

# Make a prediction
prediction = model.predict(processed_df['message'])
probability = model.predict_proba(processed_df['message'])

# Interpret the result
is_spam = prediction[0] == 1
spam_probability = probability[0][1]

print(f"Message: {message}")
print(f"Is spam: {is_spam}")
print(f"Spam probability: {spam_probability:.4f}")
```

## Project Structure

- `main.py`: Main script for training the model
- `test.py`: Script for testing the model on new messages
- `preprocessing.py`: Functions for data preprocessing
- `feature_extraction.py`: Functions for feature extraction
- `Dataset/`: Directory containing the dataset files
- `spam_classifier.joblib`: Saved trained model

## Technical Details

### Preprocessing Steps

1. Convert text to lowercase
2. Remove punctuation and numbers (preserving $ and ! as they can be indicators of spam)
3. Tokenize the text
4. Remove stop words
5. Apply stemming to reduce words to their base form
6. Rejoin tokens into a single string

### Model Details

- Algorithm: Multinomial Naive Bayes
- Feature Extraction: TF-IDF Vectorization
- Hyperparameter Tuning: Grid search with cross-validation
- Evaluation Metric: F1 score

## License

This project is for educational purposes and is based on materials from HTB Academy. Please refer to HTB Academy's terms for usage rights.

## Acknowledgments

- HTB Academy for the original materials and guidance
- UCI Machine Learning Repository for the SMS Spam Collection dataset