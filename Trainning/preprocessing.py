import requests 
import zipfile
import io
import os
import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# # Download the necessary NLTK data files
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")

# Preprocessor func
def preprocessor(df):
    """
    Preprocess the DataFrame by cleaning and tokenizing the 'message' column.
    """
    print("====================== Preprocessing Data ======================")


    # Convert all messages to lowercase
    print("Converting messages to lowercase...")
    nltk.download('punkt')  # Ensure punkt tokenizer is downloaded
    nltk.download('punkt_tab')  # Ensure punkt_tab tokenizer is downloaded
    df['message'] = df['message'].str.lower()

    # Remove punctuations and numbers, excluding $ and !
    print("Removing punctuations and numbers, excluding $ and !...")
    df['message'] = df['message'].apply(lambda row: re.sub(r'[^a-z\s$!]', "", row))

    # Tokenize the messages [i.e: split each message into words]
    print("Tokenizing messages...")
    df['message'] = df['message'].apply(word_tokenize)

    # Removing stop words from tokens
    print("Removing stop words...")
    nltk.download('stopwords')  # Ensure stopwords are downloaded
    stop_words = set(stopwords.words('english'))  # Define a set of English stop words
    df['message'] = df['message'].apply(lambda words: [word for word in words if word not in stop_words])

    # Stemming the words to reduce words to their base form
    print
    stemmer = PorterStemmer()
    df['message'] = df['message'].apply(lambda words: [stemmer.stem(word) for word in words])

    # Rejoining tokens back into single string for feature extraction
    print("Rejoining tokens into single string...")
    df['message'] = df['message'].apply(lambda words: " ".join(words))
    
    return df

def dataset(input):

    url_pattern = r'(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    
    # Check if the input is a valid URL
    if re.match(url_pattern, input):
        """
        Download and extract the dataset from the given URL.
        """
        print("Downloading dataset...")
        response = requests.get(input)
        

        if response.status_code == 200:
            print("Dataset downloaded successfully")
            with zipfile.ZipFile(io.BytesIO(response.content)) as dataset_zip:
                dataset_zip.extractall("Dataset")
                print("Dataset extracted")

                # load dataset into dataframe
                print("Loading dataset into DataFrame...")
                path = "../Dataset/SMSSpamCollection"
                if not os.path.exists(path):
                    print("Dataset file not found. Please check the extraction.")
                    exit(1)
                
                df = pd.read_csv(path, header=None,names = ['label','message'], sep='\t')
                print("Dataset loaded into DataFrame")
                return df
    
    # Check if the input is a local file path
    elif os.path.exists(input):
        """
        Load dataset from a local file path.
        """
        print("Loading dataset from local file...")
        df = pd.read_csv(input, header=None, names=['label', 'message'], sep='\t')
        print("Dataset loaded into DataFrame")
        return df
    
    else:
        print("Failed to download dataset")
        exit(1)



def data_inspection(df):
    """
    Inspect the DataFrame before processing.
    """
    print("====================== Data Sample ======================")
    print(df.head())
    print("====================== Describe Data ======================")
    print(df.describe())
    print("====================== Data Info ======================")
    print(df.info())
    print("====================== Data Cleaning ======================")
    print(df.isnull().sum())        
    """ 
    ==================================================================
    | Need to Update what to do with null values                     |
    ==================================================================
    """ 
    
    print("checking for duplicates...")
    print(f"{df.duplicated().sum()} duplicates found")
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
        print("Duplicates removed")
        print(f"{len(df)} rows after removing duplicates")

    return df

# # Download dataset
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
# response = requests.get(url)

# # Check if the request is successful
# if response.status_code == 200:
#     print("Dataset downloaded successfully")

#     # unzip the dataset
#     with zipfile.ZipFile(io.BytesIO(response.content)) as dataset_zip:
#         dataset_zip.extractall("Dataset")
#         print("Dataset extracted")
    
#     # Confirm the contents of the extracted folder
#     extracted_files = os.listdir("Dataset")
#     print(f"Extracted files: {extracted_files}")

#     # load dataset into a pandas DataFrame
#     path = "Dataset/SMSSpamCollection"
#     df = pd.read_csv(path, sep='\t', header=None, names=['label', 'message'])

#     # Inspect data before processing 
#     print("====================== Data Sample ======================")
#     print(df.head())
#     print("====================== Describe Data=====================")
#     print(df.describe())
#     print("====================== Data Info ======================")
#     print(df.info())

#     # Begin data cleaning / preprocessing
#     print("====================== Data Cleaning ======================")
    
#     print(df.isnull().sum())
#     print("====================== Check for duplicates ======================")
#     print(f"{df.duplicated().sum()} duplicates found")
#     if df.duplicated().sum() >0:
#         df = df.drop_duplicates()
#         print("Duplicates removed")
#         print(f"{len(df)}, rows after removing duplicates ")



#     # Convert all mmessages to lowercase
#     df['message'] = df['message'].str.lower()

#     # Remove punctuations and numbers, excluding $ and !
#     df['message'] = df['message'].apply(lambda row: re.sub(r'[^a-z\s$!]',"", row))
#     print("======================== Data After Removing Punctuations and Numbers ======================")
#     print([df['message'].head()])

#     # Tokenize the messages [i.e: split each message into words]
#     df['message'] = df['message'].apply(word_tokenize)
#     print("======================== Data After Tokenization ======================")
#     print(df.head())

#     # removing stop words from tookens
#     stop_words = set(stopwords.words('english'))  # Define a set of English stop words
#     df['message'] = df['message'].apply(lambda words: [ word for word in words if word not in stop_words])
#     print("======================== Data After Removing Stop Words ======================")
#     print(df.head())

#     # Stemming the words to reduce words to their base form
#     stemmer = PorterStemmer()
#     df['message'] = df['message'].apply(lambda words: [stemmer.stem(word) for word in words])
#     print("======================== Data After Stemming ======================")
#     print(df.head())   # After stemming, tokens focus on their base/root word forms

#     # Rejoining tokens back into single string for feature extraction
#     df['message'] = df['message'].apply(lambda words: "".join(words))
#     print("======================== Data After Rejoining Tokens ======================")
#     print(df.head())








# else:
#     print("Failed to download dataset")
#     exit(1)

if __name__ == "__main__":
    #Download dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    df = dataset(url)

    #inspect datset
    df = data_inspection(df)

    #preprocess dataset
    df = preprocessor(df)