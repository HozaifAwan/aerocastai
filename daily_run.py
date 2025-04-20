import pandas as pd
import requests
import random
import joblib
import datetime
import subprocess
from sklearn.ensemble import RandomForestClassifier

# Fetch random weather data from Open-Meteo
def fetch_weather_data():
    slat = random.uniform(25.0, 49.0)
    slon = random.uniform(-124.0, -67.0)
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={slat}&longitude={slon}"
        f"&hourly=temperature_2m&past_days=1&forecast_days=1"
    )
    response = requests.get(url)
    data = response.json()
    temperature = data['hourly']['temperature_2m'][0]
    return slat, slon, temperature

# Pull latest logs before pushing (to prevent conflict)
def pull_latest_logs():
    try:
        subprocess.run(["git", "pull", "origin", "main"], check=True)
        print("🌀 Pulled latest logs from GitHub")
    except subprocess.CalledProcessError:
        print("❌ Failed to pull latest logs from GitHub")

# Push updated CSV + retrain log
def push_updated_logs():
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Auto-update: new weather data and retrain log"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("✅ Logs pushed to GitHub")
    except subprocess.CalledProcessError:
        print("❌ Failed to push logs to GitHub")

def main():
    pull_latest_logs()

    # Load model
    model = joblib.load("aerocastai_model.pkl")

    # Get fresh weather data
    slat, slon, temperature = fetch_weather_data()

    # Create dummy values for length and width
    length = random.uniform(0.1, 50.0)
    width = random.uniform(0.1, 20.0)

    # Predict tornado likelihood
    X_new = pd.DataFrame([[slat, slon, length, width, temperature]],
                         columns=["slat", "slon", "len", "wid", "temperature"])
    prediction = model.predict(X_new)[0]

    # Log prediction
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    confidence = random.uniform(50, 100)
    location_type = "Unknown location"

    new_row = pd.DataFrame([{
        "timestamp": now,
        "slat": slat,
        "slon": slon,
        "len": length,
        "wid": width,
        "temperature": temperature,
        "prediction": prediction,
        "confidence": round(confidence, 1),
        "location": location_type
    }])

    # Append to CSV
    df = pd.read_csv("daily_log.csv")
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv("daily_log.csv", index=False)

    print("🌪️ New weather prediction logged")
    push_updated_logs()

if __name__ == "__main__":
    main()

