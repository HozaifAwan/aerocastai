from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import requests
import joblib
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

MODEL_PATH = "aerocastai_model.pkl"
USER_LOG = "user_log.csv"

@app.route('/user-log', methods=['POST'])
def user_log():
    try:
        location_name = request.json.get("location", "")
        if not location_name:
            return jsonify({"error": "Location required"}), 400

        # Geocode location to lat/lon
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_name}&count=1"
        geo_res = requests.get(geo_url).json()
        if "results" not in geo_res or len(geo_res["results"]) == 0:
            return jsonify({"error": "Location not found"}), 404

        geo = geo_res["results"][0]
        lat = round(geo["latitude"], 4)
        lon = round(geo["longitude"], 4)
        resolved_location = f"{geo['name']}, {geo.get('admin1', '')}".strip()

        # Get Open-Meteo data
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,dew_point_2m,relative_humidity_2m,precipitation,cloudcover,"
            f"surface_pressure,convective_available_potential_energy,lifted_index,"
            f"wind_speed_10m,wind_gusts_10m&forecast_days=1&timezone=auto"
        )
        weather_data = requests.get(weather_url).json()
        if "hourly" not in weather_data:
            return jsonify({"error": "Weather data not found"}), 500

        hourly = weather_data["hourly"]
        idx = 0  # Latest hour

        data = {
            "lat": lat,
            "lon": lon,
            "temperature": hourly["temperature_2m"][idx],
            "dew_point_2m": hourly["dew_point_2m"][idx],
            "relative_humidity_2m": hourly["relative_humidity_2m"][idx],
            "precipitation": hourly["precipitation"][idx],
            "cloudcover": hourly["cloudcover"][idx],
            "surface_pressure": hourly["surface_pressure"][idx],
            "wind_speed_10m": hourly["wind_speed_10m"][idx],
            "wind_gusts_10m": hourly["wind_gusts_10m"][idx],
            "convective_available_potential_energy": hourly["convective_available_potential_energy"][idx],
            "lifted_index": hourly["lifted_index"][idx],
            "location": resolved_location
        }

        # Predict
        model = joblib.load(MODEL_PATH)
        X = pd.DataFrame([data])
        prediction = int(model.predict(X)[0])
        confidence = round(model.predict_proba(X)[0][prediction] * 100, 2)

        # Log entry
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **data,
            "prediction": prediction,
            "confidence": confidence
        }

        log_df = pd.DataFrame([entry])
        log_df.to_csv(USER_LOG, mode='a', header=not os.path.exists(USER_LOG), index=False)

        return jsonify(entry)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



