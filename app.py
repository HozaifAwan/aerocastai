# --- app.py (Backend Flask API) ---
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import requests
from datetime import datetime

app = Flask(__name__)
CORS(app)

MODEL_PATH = "aerocastai_model.pkl"
USER_LOG = "user_log.csv"

def fetch_location_data(location):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
    geo_res = requests.get(geo_url)
    geo_data = geo_res.json()
    if not geo_data.get("results"):
        return None, "Location not found."

    geo = geo_data["results"][0]
    lat = geo["latitude"]
    lon = geo["longitude"]
    full_name = ", ".join([geo["name"], geo.get("admin1", ""), geo.get("country", "")])

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&hourly=wind_speed_10m,wind_gusts_10m,temperature_2m,dew_point_2m,"
        f"relative_humidity_2m,precipitation,cloudcover,surface_pressure,"
        f"cape,lifted_index&forecast_days=1&timezone=auto"
    )
    weather_res = requests.get(weather_url).json()
    if "hourly" not in weather_res or not weather_res["hourly"]["time"]:
        return None, "Weather data not found."

    idx = 0
    data = {
        "lat": lat,
        "lon": lon,
        "temperature": weather_res["hourly"]["temperature_2m"][idx],
        "dew_point_2m": weather_res["hourly"]["dew_point_2m"][idx],
        "relative_humidity_2m": weather_res["hourly"]["relative_humidity_2m"][idx],
        "precipitation": weather_res["hourly"]["precipitation"][idx],
        "cloudcover": weather_res["hourly"]["cloudcover"][idx],
        "surface_pressure": weather_res["hourly"]["surface_pressure"][idx],
        "wind_speed_10m": weather_res["hourly"]["wind_speed_10m"][idx],
        "wind_gusts_10m": weather_res["hourly"]["wind_gusts_10m"][idx],
        "convective_available_potential_energy": weather_res["hourly"]["convective_available_potential_energy"][idx],
        "lifted_index": weather_res["hourly"]["lifted_index"][idx]
    }
    return (full_name, data), None

@app.route("/user-log", methods=["POST"])
def handle_user_log():
    req_data = request.get_json()
    location_input = req_data.get("location")
    if not location_input:
        return jsonify({"error": "Missing location"}), 400

    result, err = fetch_location_data(location_input)
    if err:
        return jsonify({"error": err}), 400

    location_name, features = result
    model = joblib.load(MODEL_PATH)
    X_input = pd.DataFrame([features])
    prediction = int(model.predict(X_input)[0])
    confidence = round(model.predict_proba(X_input)[0][prediction] * 100, 2)

    log_row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **features,
        "prediction": prediction,
        "confidence": confidence,
        "location": location_name
    }

    df = pd.DataFrame([log_row])
    df.to_csv(USER_LOG, mode="a", header=not os.path.exists(USER_LOG), index=False)

    return jsonify(log_row)

if __name__ == "__main__":
    app.run(debug=True)


