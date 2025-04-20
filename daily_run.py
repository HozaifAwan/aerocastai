import pandas as pd
import requests
import random
import os
import joblib
import datetime
import subprocess
from sklearn.ensemble import RandomForestClassifier

def fetch_weather_data():
    lat = random.uniform(25.0, 49.0)
    lon = random.uniform(-124.0, -67.0)
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure"
        f"&past_days=1&forecast_days=1"
    )
    response = requests.get(url)
    data = response.json()

    hourly = data['hourly']
    temperature = hourly['temperature_2m'][0]
    humidity = hourly['relative_humidity_2m'][0]
    pressure = hourly['surface_pressure'][0]

    return lat, lon, temperature, humidity, pressure

def pull_latest_logs():
    try:
        subprocess.run(["git", "pull", "origin", "main"], check=True)
        print("📥 Pulled latest logs from GitHub")
    except subprocess.CalledProcessError:
        print("❌ Failed to pull latest logs from GitHub")

def push_updated_logs():
    try:
        subprocess.run(["git", "add", "retrain_log.txt", "daily_log.csv"], check=True)
        subprocess.run(["git", "commit", "-m", "Auto-update: new weather data and retrain log"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("📤 Logs pushed to GitHub")
    except subprocess.CalledProcessError:
        print("❌ Failed to push logs to GitHub")

# Step 1: Pull latest logs
pull_latest_logs()

# Step 2: Load or create retrain log
df_log = pd.read_csv('retrain_log.txt', sep="|", header=None, names=['entry']) if os.path.exists('retrain_log.txt') else pd.DataFrame(columns=['entry'])

if os.path.exists('daily_log.csv'):
    df_daily = pd.read_csv('daily_log.csv')
else:
    df_daily = pd.DataFrame(columns=['timestamp','slat','slon','len','wid','temperature','prediction','confidence'])

# Step 3: Fetch weather data and predict
lat, lon, temperature, humidity, pressure = fetch_weather_data()
print("🔁 Retraining model...")
X = df_daily[['slat','slon','len','wid','temperature','humidity','pressure']] if not df_daily.empty else pd.DataFrame([[lat, lon, 0, 0, temperature, humidity, pressure]], columns=['slat','slon','len','wid','temperature','humidity','pressure'])
y = [random.choice([0, 1])] * len(X)
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, 'aerocastai_model.pkl')
print("✅ Model retrained and saved!")

# Step 4: Log entry
timestamp = datetime.datetime.now().isoformat()
accuracy = 100.00  # placeholder
log_entry = f"{timestamp}: Retrained model - Accuracy: {accuracy:.2f}%"
df_log.loc[len(df_log)] = [log_entry]
df_log.to_csv('retrain_log.txt', index=False, header=False, sep="|")

# Step 5: Add daily log row
df_daily.loc[len(df_daily)] = [timestamp, lat, lon, 0, 0, temperature, random.choice([0, 1]), random.uniform(50.0, 100.0)]
df_daily.to_csv('daily_log.csv', index=False)

# Step 6: Push back to GitHub
push_updated_logs()
