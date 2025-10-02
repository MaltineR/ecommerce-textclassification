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

def plot_category_distribution(df):
    counts = df['category'].value_counts()
    plt.figure(figsize=(7,5))
    counts.plot(kind='bar')
    plt.title("Number of samples per category")
    plt.ylabel("Count")
    plt.show()

plot_category_distribution(df)


def plot_wordcloud(df, category_name):
    text = ' '.join(df[df['category'] == category_name]['processed_desc'])
    wc = WordCloud(background_color='white', max_words=50)
    plt.figure(figsize=(7,7))
    plt.imshow(wc.generate(text))
    plt.axis('off')
    plt.title(category_name)
    plt.show()

for cat in df['category'].unique():
    plot_wordcloud(df, cat)


vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['processed_desc'])
y = df['category_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

print(f'Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}')

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'LogisticRegression': LogisticRegression(max_iter=500)
}

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
   
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
   