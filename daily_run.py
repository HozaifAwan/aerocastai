import pandas as pd
import requests
import random
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

# Function to fetch weather data from Open-Meteo for a random US location
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

# Load or create retrain log
log_path = 'retrain_log.txt'
if os.path.exists(log_path):
    df_log = pd.read_csv(log_path)
else:
    df_log = pd.DataFrame(columns=['lat', 'lon', 'temperature', 'humidity', 'pressure', 'tornado'])

# Fetch new weather data
lat, lon, temperature, humidity, pressure = fetch_weather_data()
tornado = random.choice([0, 1])  # Randomly assign label for now

# Append new row to log
df_log.loc[len(df_log)] = [lat, lon, temperature, humidity, pressure, tornado]
df_log.to_csv(log_path, index=False)

# Retrain the model
print("🔁 Retraining model...")
X = df_log[['lat', 'lon', 'temperature', 'humidity', 'pressure']]
y = df_log['tornado']
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'aerocastai_model.pkl')
print("✅ Model retrained and saved!")
