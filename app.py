# app.py (Flask backend for AeroCastAI)
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS  # if CORS is needed

app = Flask(__name__)
# Enable CORS for all routes (optional, needed if frontend is on a different domain)
CORS(app)

# Load the trained model once at startup
model = joblib.load("aerocastai_model.pkl")

@app.route("/")
def home():
    return "✅ AeroCastAI backend is live."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)  # parse JSON request body
        # Required features from frontend (12 total)
        expected_features = [
            "lat", "lon",
            "wind_speed_10m", "wind_gusts_10m", "temperature",
            "dew_point_2m", "relative_humidity_2m", "precipitation",
            "cloudcover", "surface_pressure",
            "convective_available_potential_energy", "lifted_index"
        ]
        # Verify all features are present in the JSON
        if not all(feat in data and data[feat] is not None for feat in expected_features):
            return jsonify({"error": "Missing one or more required fields."}), 400

        # Construct DataFrame for model input
        input_df = pd.DataFrame([[data[feat] for feat in expected_features]], columns=expected_features)
        # Make prediction
        pred = model.predict(input_df)[0]
        probas = model.predict_proba(input_df)[0]
        confidence = round(float(probas[int(pred)] * 100), 2)

        return jsonify({
            "prediction": int(pred),
            "confidence": confidence
        })
    except Exception as e:
        # Catch any unexpected errors
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False)

