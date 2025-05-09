from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("aerocastai_model.pkl")

@app.route("/")
def home():
    return "✅ AeroCastAI backend is live."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Required features from frontend
        features = [
            "lat", "lon", "wind_speed_10m", "wind_gusts_10m", "temperature",
            "dew_point_2m", "relative_humidity_2m", "precipitation",
            "cloudcover", "surface_pressure",
            "convective_available_potential_energy", "lifted_index"
        ]

        # Ensure all features are present
        if not all(feat in data for feat in features):
            return jsonify({"error": "Missing one or more required fields."}), 400

        input_df = pd.DataFrame([[data[feat] for feat in features]], columns=features)

        pred = model.predict(input_df)[0]
        probas = model.predict_proba(input_df)[0]
        confidence = round(probas[int(pred)] * 100, 2)

        return jsonify({
            "prediction": int(pred),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
