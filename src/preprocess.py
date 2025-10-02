import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')
nltk.download('punkt')

StopWords = set(stopwords.words('english'))
stemmer = PorterStemmer()

def load_data(file_path='data/ecommerceDataset.csv'):
    # Load CSV without headers
    df = pd.read_csv(file_path, names=['category', 'description'], header=None)
    df.dropna(inplace=True)  # Drop missing descriptions
    df['category'] = df['category'].str.replace('&', '_').str.replace(' ', '_')
    return df