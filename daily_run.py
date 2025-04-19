import pandas as pd
import requests
import joblib
import datetime

print("✅ [START] Script started at", datetime.datetime.now())

# Load the saved model
print("📦 Loading model...")
model = joblib.load("aerocastai_model.pkl")

# Define coordinates
slat = 35.4676
slon = -97.5164

# Fetch weather data
print("🌐 Fetching weather data...")
response = requests.get(
    f"https://api.open-meteo.com/v1/forecast?latitude={slat}&longitude={slon}&hourly=temperature_2m,relative_humidity_2m,surface_pressure"
)

if response.status_code != 200:
    print("❌ Failed to fetch data from API:", response.status_code)
    exit()

data = response.json()
hourly = data.get("hourly", {})

temperature = hourly["temperature_2m"][0]
humidity = hourly["relative_humidity_2m"][0]
pressure = hourly["surface_pressure"][0]

# Create dataframe for prediction
df = pd.DataFrame(
    [[slat, slon, 0, 0, temperature, humidity, pressure]],
    columns=["slat", "slon", "len", "wid", "temperature", "humidity", "pressure"]
)

# Predict
print("🧠 Running prediction...")
prediction = model.predict(df)[0]
confidence = max(model.predict_proba(df)[0])

# Save results
print("✍️ Logging to CSV...")
df["prediction"] = prediction
df["confidence"] = confidence

# Append to log
df.to_csv("daily_log.csv", mode="a", header=False, index=False)

print("✅ [DONE] Script finished at", datetime.datetime.now())
