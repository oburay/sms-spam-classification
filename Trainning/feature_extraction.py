from sklearn.feature_extraction.text import CountVectorizer


def feature_extraction(df):
    """
    Extract features from the DataFrame using CountVectorizer.
    """
    print("====================== Feature Extraction ======================")

    # Initialize CountVectorizer with bigrams, min_df, and max_df to focus on relevant terms
    vectorizer = CountVectorizer(min_df=1, max_df=0.9, ngram_range=(1,2))

    # Fit and transform the 'message' column to create a document-term matrix
    print("Fitting and transforming the 'message' column...")
    X = vectorizer.fit_transform(df['message'])

    # Labels (target variable)
    y = df['label'].apply(lambda x: 1 if x == 'spam' else 0)

    # Convert the sparse matrix to a dense format and create a DataFrame
    # feature_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # return feature_df, vectorizer
    print("Feature extraction completed.")
    return X, y, vectorizer
    