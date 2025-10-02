from flask import Flask, request, jsonify
from src.predict import predict_category

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if text:
        category = predict_category(text)
        return jsonify({'category': category})
    return jsonify({'error': 'No text provided'}), 400

if __name__ == "__main__":
    app.run(debug=True)
