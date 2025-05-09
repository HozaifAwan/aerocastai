from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import datetime
import joblib

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("aerocastai_model.pkl")

USER_LOG_FILE = "user_log.csv"

@app.route('/')
def home():
    return "✅ AeroCastAI backend is live."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        required_fields = [
            "lat", "lon", "temperature", "dew_point_2m", "relative_humidity_2m",
            "precipitation", "cloudcover", "surface_pressure", "wind_speed_10m",
            "wind_gusts_10m", "convective_available_potential_energy", "lifted_index"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        features = [[
            data["lat"], data["lon"], data["temperature"], data["dew_point_2m"],
            data["relative_humidity_2m"], data["precipitation"], data["cloudcover"],
            data["surface_pressure"], data["wind_speed_10m"], data["wind_gusts_10m"],
            data["convective_available_potential_energy"], data["lifted_index"]
        ]]

        prediction = int(model.predict(features)[0])
        confidence = round(model.predict_proba(features)[0][prediction] * 100, 2)

        log_entry = {
            **data,
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": datetime.datetime.now().isoformat()
        }

        # Append to user_log.csv
        df = pd.DataFrame([log_entry])
        if not os.path.exists(USER_LOG_FILE):
            df.to_csv(USER_LOG_FILE, index=False)
        else:
            df.to_csv(USER_LOG_FILE, mode='a', header=False, index=False)

        return jsonify(log_entry)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


