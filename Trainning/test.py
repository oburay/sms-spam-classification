from main import *

"""
===========Test the model with unseen data===========
"""

# load the data 
raw_test_data = dataset("Dataset/test_message")

# Define test parameters
sample_size = 15
sample_method = "tail"  # Change to "head" if you want to test first messages

# Get sample data using the chosen method
if sample_method == "head":
    sample_indices = raw_test_data.head(sample_size).index
elif sample_method == "tail":
    sample_indices = raw_test_data.tail(sample_size).index
else:
    # You could add random sampling option here
    sample_indices = raw_test_data.sample(sample_size).index

# Store original messages before processing
truly_original_messages = raw_test_data.loc[sample_indices, 'message'].copy()

# Continue with processing
orginal_test_data = raw_test_data.copy()
orginal_test_data = data_inspection(orginal_test_data)
test_data = preprocessor(orginal_test_data)

# Extract processed messages using the same indices
processed_messages = test_data.loc[sample_indices, 'message']

# Transform the selected messages
X_new = best_model.named_steps['vectorizer'].transform(processed_messages)

# Make predictions
predictions = best_model.named_steps['classifier'].predict(X_new)
predict_probs = best_model.named_steps['classifier'].predict_proba(X_new)

# Display results with proper alignment
print(f"\n=========== Testing with {sample_size} {sample_method} messages ===========\n")

for i, idx in enumerate(sample_indices):
    prediction = "Spam" if predictions[i] == 1 else "Ham"
    prob_spam = predict_probs[i][1]
    prob_ham = predict_probs[i][0]
    
    print(f"Original Message: {truly_original_messages.iloc[i]}")
    print(f"Processed Message: {processed_messages.iloc[i]}")
    print(f"Prediction: {prediction}")
    print(f"Probability of Spam: {prob_spam:.4f}")
    print(f"Probability of Ham: {prob_ham:.4f}")
    print("-" * 50)