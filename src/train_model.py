import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import load_data, preprocess_dataframe


df = load_data(file_path='data/ecommerceDataset.csv')
df = preprocess_dataframe(df)


category_to_num = {cat: idx for idx, cat in enumerate(df['category'].unique())}
df['category_num'] = df['category'].map(category_to_num)