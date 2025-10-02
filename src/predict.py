import joblib
from .preprocess import preprocess_text



category_classifier= joblib.load('models/RandomForestClassifier.sav')
vectorizer = joblib.load('models/vectorizer.sav')
category_to_num = joblib.load('models/category_to_num.sav')
num_to_category = {v: k for k, v in category_to_num.items()}

def predict_category(text):
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed])
    pred_num = category_classifier.predict(vectorized)[0]
    return num_to_category[pred_num]

if __name__ == "__main__":
    sample = "Wireless Bluetooth Headphones with mic"
    print("Predicted category:", predict_category(sample))