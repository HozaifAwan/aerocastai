from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
import csv
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("aerocastai_model.pkl")
USER_LOG_FILE = "user_log.csv"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    required_fields = [
        "lat", "lon", "temperature", "dew_point_2m", "relative_humidity_2m", "precipitation",
        "cloudcover", "surface_pressure", "wind_speed_10m", "wind_gusts_10m",
        "convective_available_potential_energy", "lifted_index"
    ]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required input fields."}), 400

    input_df = pd.DataFrame([{
        "lat": data["lat"],
        "lon": data["lon"],
        "temperature": data["temperature"],
        "dew_point_2m": data["dew_point_2m"],
        "relative_humidity_2m": data["relative_humidity_2m"],
        "precipitation": data["precipitation"],
        "cloudcover": data["cloudcover"],
        "surface_pressure": data["surface_pressure"],
        "wind_speed_10m": data["wind_speed_10m"],
        "wind_gusts_10m": data["wind_gusts_10m"],
        "convective_available_potential_energy": data["convective_available_potential_energy"],
        "lifted_index": data["lifted_index"]
    }])

    prediction = int(model.predict(input_df)[0])
    confidence = round(float(model.predict_proba(input_df)[0][prediction]) * 100, 2)

    full_response = {
        **data,
        "prediction": prediction,
        "confidence": confidence
    }

    # Log to user_log.csv
    log_user_data(full_response)

    return jsonify(full_response)

def log_user_data(entry):
    file_exists = os.path.isfile(USER_LOG_FILE)
    with open(USER_LOG_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)

@app.route("/last-user-log", methods=["GET"])
def last_user_log():
    if not os.path.exists(USER_LOG_FILE):
        return jsonify({"error": "No logs found."}), 404
    df = pd.read_csv(USER_LOG_FILE)
    if df.empty:
        return jsonify({"error": "Log is empty."}), 404
    last = df.iloc[-1].to_dict()
    return jsonify(last)

if __name__ == "__main__":
    app.run(debug=True)


