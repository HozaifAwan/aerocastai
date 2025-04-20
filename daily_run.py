import joblib
import pandas as pd
import datetime
import random
import subprocess
import requests

def fetch_weather_data():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 34.05,      # example: Los Angeles
        "longitude": -118.25,
        "current": "temperature_2m,wind_speed_10m",
    }
    response = requests.get(url, params=params)
    data = response.json()

    slat = params["latitude"]
    slon = params["longitude"]
    length = round(random.uniform(1.0, 50.0), 1)
    width = round(random.uniform(1.0, 30.0), 1)
    temperature = data["current"]["temperature_2m"]

    return slat, slon, length, width, temperature

def setup_git_identity():
    try:
        subprocess.run(['git', 'config', '--global', 'user.email', 'hozaifawan@gmail.com'], check=True)
        subprocess.run(['git', 'config', '--global', 'user.name', 'Hozaif Awan'], check=True)
        subprocess.run(['git', 'remote', 'add', 'origin', 'https://github.com/HozaifAwan/aerocastai.git'], check=False)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Git identity setup warning: {e}")

def push_updated_logs():
    try:
        setup_git_identity()
        subprocess.run(['git', 'pull', 'origin', 'main'], check=True)
        print("✅ Pulled latest logs from GitHub")

        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Auto-update: new weather data and retrain log'], check=True)
        subprocess.run(['git', 'push', '-u', 'origin', 'main', '--force'], check=True)
        print("✅ Logs pushed to GitHub")
    except subprocess.CalledProcessError as e:
        print(f"❌ Git error: {e}")

def pull_latest_logs():
    try:
        subprocess.run(['git', 'pull', 'origin', 'main'], check=True)
        print("✅ Pulled latest logs from GitHub")
    except subprocess.CalledProcessError:
        print("❌ Failed to pull latest logs from GitHub")

def main():
    pull_latest_logs()

    model = joblib.load("aerocastai_model.pkl")
    slat, slon, length, width, temperature = fetch_weather_data()

    X_new = pd.DataFrame([[slat, slon, length, width, temperature]],
                         columns=["slat", "slon", "len", "wid", "temperature"])

    prediction = model.predict(X_new)[0]

    # Log prediction
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    confidence = round(random.uniform(50.0, 100.0), 2)

    new_data = pd.DataFrame([[now, slat, slon, length, width, temperature, prediction, confidence, "Unknown location"]],
                            columns=["timestamp", "slat", "slon", "len", "wid", "temperature", "prediction", "confidence", "location"])

    try:
        existing_data = pd.read_csv("daily_log.csv")
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        updated_data = new_data

    updated_data.to_csv("daily_log.csv", index=False)
    print("🌦️ New weather prediction logged")

    push_updated_logs()

if __name__ == "__main__":
    main()
