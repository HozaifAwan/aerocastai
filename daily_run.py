import schedule
import joblib
import pandas as pd
from datetime import datetime
from retrain_model import model
import os
import subprocess
import requests
import random

# Fetch random weather data from Open-Meteo
latitude = round(random.uniform(-90, 90), 4)
longitude = round(random.uniform(-180, 180), 4)
location_name = "Unknown location"

# Fetch live weather data
url = (
    f"https://api.open-meteo.com/v1/forecast"
    f"?latitude={latitude}&longitude={longitude}"
    f"&hourly=wind_speed_10m,wind_gusts_10m,temperature_2m"
    f"&forecast_days=1&timezone=America%2FChicago"
)
response = requests.get(url).json()

# Grab latest weather details
wind_speed = response["hourly"]["wind_speed_10m"][0]
wind_gusts = response["hourly"]["wind_gusts_10m"][0]
temperature = response["hourly"]["temperature_2m"][0]

# Prediction
input_data = pd.DataFrame([{
    "slat": latitude,
    "slon": longitude,
    "len": wind_speed,
    "wid": wind_gusts,
    "temperature": temperature
}])

# Ensure binary prediction (0 or 1)
prediction = int(model.predict(input_data)[0] > 0.5)

# Get probabilities correctly for binary classification
probas = model.predict_proba(input_data)[0]

# Correct confidence handling for binary predictions
confidence = round(probas[prediction] * 100, 2)

# Log data consistently
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_entry = {
    "timestamp": timestamp,
    "slat": latitude,
    "slon": longitude,
    "len": wind_speed,
    "wid": wind_gusts,
    "temperature": temperature,
    "prediction": prediction,
    "confidence": confidence,
    "location": location_name
}

log_df = pd.DataFrame([log_entry])
log_df.to_csv("daily_log.csv", mode="a", header=not os.path.exists("daily_log.csv"), index=False)

print(f"🌦️ Logged new prediction: {log_entry}")

# Retrain the model
df = pd.read_csv("daily_log.csv").dropna()
if not df.empty:
    X = df[["slat", "slon", "len", "wid", "temperature"]]
    y = df["prediction"]

    new_model = model.fit(X, y)
    joblib.dump(new_model, "aerocastai_model.pkl")

    with open("retrain_log.txt", "a") as f:
        f.write(f"{timestamp}: Retrained model - Accuracy: {new_model.score(X, y)*100:.2f}%\n")

    print("✅ Model retrained and saved")
else:
    print("⚠️ No data available for retraining.")

# GitHub Auto Push (no token in the file)
try:
    subprocess.run(["git", "config", "--global", "user.email", "hozaifawan@render.com"], check=True)
    subprocess.run(["git", "config", "--global", "user.name", "HozaifAwan"], check=True)
    subprocess.run(["git", "add", "daily_log.csv", "retrain_log.txt", "aerocastai_model.pkl"], check=True)
    subprocess.run(["git", "commit", "-m", f"Auto-update: weather data & retrained model ({timestamp})"], check=True)

    subprocess.run(["git", "push", "-u", "origin", "main", "--force"], check=True)

    print("✅ Successfully pushed logs and model to GitHub")
except subprocess.CalledProcessError as e:
    print(f"❌ Git push error: {e}")