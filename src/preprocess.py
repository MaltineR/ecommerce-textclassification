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
    df = pd.read_csv(file_path, names=['category', 'description'], header=None)
    df.dropna(inplace=True)
    df['category'] = df['category'].str.replace('&', '_').str.replace(' ', '_')
    return df
def preprocess_text(text):
    text_cleaned = re.sub(r"[^\w\s']", '', text).lower()
    tokens = word_tokenize(text_cleaned)
    tokens = [stemmer.stem(w) for w in tokens if w not in StopWords and w not in string.punctuation]
    return ' '.join(tokens)

def preprocess_dataframe(df):
    df['processed_desc'] = df['description'].apply(preprocess_text)
    return df