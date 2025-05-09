from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("aerocastai_model.pkl")

@app.route("/")
def home():
    return "✅ AeroCastAI backend is live."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expected input features in exact order:
        features = [
            "lat", "lon", "wind_speed_10m", "wind_gusts_10m", "temperature",
            "dew_point_2m", "relative_humidity_2m", "precipitation",
            "cloudcover", "surface_pressure", "convective_available_potential_energy",
            "lifted_index"
        ]

        input_df = pd.DataFrame([[data[feat] for feat in features]], columns=features)

        pred = model.predict(input_df)[0]
        probas = model.predict_proba(input_df)[0]
        confidence = round(probas[int(pred)] * 100, 2)

        return jsonify({
            "prediction": int(pred),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

