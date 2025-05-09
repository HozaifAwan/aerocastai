from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("aerocastai_model.pkl", "rb"))

@app.route('/')
def home():
    return '✅ AeroCastAI backend is live.'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Adjust this based on your real feature names
    try:
        features = np.array([[
            data["temperature"],
            data["humidity"],
            data["pressure"],
            data["wind_speed"]
        ]])
    except KeyError:
        return jsonify({"error": "Missing one or more required fields."}), 400

    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
