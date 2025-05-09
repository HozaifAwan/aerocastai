from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("aerocastai_model.pkl")

@app.route("/")
def home():
    return "✅ AeroCastAI backend is live."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = [
            "lat", "lon", "wind_speed_10m", "wind_gusts_10m", "temperature",
            "dew_point_2m", "relative_humidity_2m", "precipitation",
            "cloudcover", "surface_pressure",
            "convective_available_potential_energy", "lifted_index"
        ]

        input_df = pd.DataFrame([[data[feat] for feat in features]], columns=features)

        pred = model.predict(input_df)[0]
        try:
            probas = model.predict_proba(input_df)[0]
            confidence = round(max(probas) * 100, 2)
        except AttributeError:
            confidence = 100.0  # fallback if model doesn't support predict_proba

        return jsonify({
            "prediction": int(pred),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run()


