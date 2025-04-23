import pandas as pd
import requests
import joblib
import os
import random
import base64
from datetime import datetime
from retrain_model import run as retrain_model

# ====== CONFIG ======
CSV_FILE = "daily_log.csv"
HEADER = [
    "timestamp", "lat", "lon", "temperature", "dew_point_2m", "relative_humidity_2m",
    "precipitation", "cloudcover", "surface_pressure", "convective_available_potential_energy",
    "lifted_index", "wind_speed_10m", "wind_gusts_10m", "prediction", "confidence", "location"
]

# ====== LOG DATA FUNCTION ======
def log_random_weather():
    lat = round(random.uniform(-90, 90), 4)
    lon = round(random.uniform(-180, 180), 4)
    location = "Unknown location"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,dew_point_2m,relative_humidity_2m,precipitation,cloudcover"
        f",surface_pressure,convective_available_potential_energy,lifted_index,wind_speed_10m,wind_gusts_10m"
        f"&forecast_days=1&timezone=auto"
    )

    try:
        response = requests.get(url)
        data = response.json()
        hourly = data.get("hourly", {})

        def get_value(key):
            return hourly.get(key, [None])[0]

        entry = {
            "timestamp": timestamp,
            "lat": lat,
            "lon": lon,
            "temperature": get_value("temperature_2m"),
            "dew_point_2m": get_value("dew_point_2m"),
            "relative_humidity_2m": get_value("relative_humidity_2m"),
            "precipitation": get_value("precipitation"),
            "cloudcover": get_value("cloudcover"),
            "surface_pressure": get_value("surface_pressure"),
            "convective_available_potential_energy": get_value("convective_available_potential_energy"),
            "lifted_index": get_value("lifted_index"),
            "wind_speed_10m": get_value("wind_speed_10m"),
            "wind_gusts_10m": get_value("wind_gusts_10m"),
            "prediction": None,
            "confidence": None,
            "location": location
        }

        input_df = pd.DataFrame([entry])
        model = joblib.load("aerocastai_model.pkl")
        input_features = input_df[[
            "lat", "lon", "wind_speed_10m", "wind_gusts_10m", "temperature",
            "dew_point_2m", "relative_humidity_2m", "precipitation", "cloudcover",
            "surface_pressure", "convective_available_potential_energy", "lifted_index"
        ]].astype(float)

        prediction = model.predict(input_features)[0]
        probas = model.predict_proba(input_features)[0]
        confidence = round(probas[int(prediction)] * 100, 2)

        input_df["prediction"] = int(prediction)
        input_df["confidence"] = confidence

        input_df.to_csv(CSV_FILE, mode="a", header=not os.path.exists(CSV_FILE), index=False)
        print(f"✅ Logged random data entry at {timestamp}")
    except Exception as e:
        print(f"❌ Failed to log weather data: {e}")

# ====== GITHUB PUSH FUNCTION ======
def upload_to_github(filepath, message, repo, token):
    filename = os.path.basename(filepath)
    url = f"https://api.github.com/repos/{repo}/contents/{filename}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    with open(filepath, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    # Get file SHA if already exists
    sha = None
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        sha = r.json().get("sha")

    data = {"message": message, "content": content}
    if sha:
        data["sha"] = sha

    r = requests.put(url, headers=headers, json=data)
    if r.status_code in [200, 201]:
        print(f"📤 Uploaded {filename} to GitHub")
    else:
        print(f"❌ GitHub upload failed for {filename}: {r.status_code} {r.text}")

# ====== MAIN EXECUTION ======
def run():
    log_random_weather()
    retrain_model()

    token = os.environ.get("GH_TOKEN")
    repo = "HozaifAwan/aerocastai"
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if token:
        upload_to_github("daily_log.csv", f"Update log {timestamp}", repo, token)
        upload_to_github("aerocastai_model.pkl", f"Update model {timestamp}", repo, token)
    else:
        print("❌ GitHub token not set in environment (GH_TOKEN)")

if __name__ == "__main__":
    run()
