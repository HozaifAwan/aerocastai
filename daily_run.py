import pandas as pd
import requests
import joblib
import time
import schedule
from datetime import datetime
from urllib.parse import quote
import random

location_name = "Unknown location"  # 🧠 GLOBAL default value

def run_aerocastai():
    global location_name

    try:
        is_manual = True
        input_mode = input("🔧 Enter mode (1 = City/State/Country, 2 = Latitude/Longitude): ").strip()

        if input_mode == "1":
            user_location = input("🌍 Enter location (e.g. Houston, Texas, United States): ").strip()
            if user_location:
                encoded_location = quote(user_location)
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_location}&count=1"
                geo_response = requests.get(geo_url).json()
                if "results" in geo_response and len(geo_response["results"]) > 0:
                    result = geo_response["results"][0]
                    latitude = result["latitude"]
                    longitude = result["longitude"]
                    location_name = ", ".join([x for x in [result.get("name", ""), result.get("admin1", ""), result.get("country", "")] if x])
                else:
                    raise ValueError("Location not found.")
            else:
                raise ValueError("Empty input.")

        elif input_mode == "2":
            latitude = float(input("🌐 Enter latitude: "))
            longitude = float(input("🌐 Enter longitude: "))
            location_name = "Manual coordinates"
        else:
            raise ValueError("Invalid mode selected.")

    except Exception as e:
        print(f"❌ Location input failed: {e}. Using random coordinates for automated run.")
        is_manual = False
        latitude = round(random.uniform(-90, 90), 4)
        longitude = round(random.uniform(-180, 180), 4)
        location_name = "Unknown location"

    print(f"\n🕒 Running AEROCASTAI at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&hourly=wind_speed_10m,wind_gusts_10m,temperature_2m"
        f"&forecast_days=1&timezone=America%2FChicago"
    )
    response = requests.get(url)
    data = response.json()

    latest_index = 0
    wind_speed = data["hourly"]["wind_speed_10m"][latest_index]
    wind_gusts = data["hourly"]["wind_gusts_10m"][latest_index]
    temperature = data["hourly"]["temperature_2m"][latest_index]

    try:
        if wind_speed is None or wind_gusts is None:
            print("🌐 Fetching NOAA fallback data...")
            noaa_url = f"https://api.weather.gov/points/{latitude},{longitude}"
            noaa_response = requests.get(noaa_url).json()
            forecast_url = noaa_response['properties']['forecastHourly']
            forecast_data = requests.get(forecast_url).json()
            first_period = forecast_data['properties']['periods'][0]
            wind_speed = float(first_period['windSpeed'].split()[0]) / 2.237
            wind_gusts = wind_speed + 5.0
    except:
        print("⚠️ NOAA fallback failed.")

    if location_name == "Unknown location":
        try:
            reverse_geo_url = f"https://geocoding-api.open-meteo.com/v1/reverse?latitude={latitude}&longitude={longitude}"
            reverse_geo_response = requests.get(reverse_geo_url).json()
            if "results" in reverse_geo_response and len(reverse_geo_response["results"]) > 0:
                reverse = reverse_geo_response["results"][0]
                location_name = ", ".join([x for x in [reverse.get("name", ""), reverse.get("admin1", ""), reverse.get("country", "")] if x])
        except:
            pass

    print(f"\n📍 Location: {location_name}")
    print("\n🌤️ Live Weather Snapshot:")
    print(f"💨 Wind Speed: {wind_speed} m/s")
    print(f"🌪️ Gusts: {wind_gusts} m/s")
    print(f"🌡️ Temperature: {temperature}°C")

    model = joblib.load("aerocastai_model.pkl")

    input_data = pd.DataFrame([{
        "slat": latitude,
        "slon": longitude,
        "len": wind_speed,
        "wid": wind_gusts
    }])

    prediction = model.predict(input_data)[0]
    probas = model.predict_proba(input_data)
    confidence = probas[0][int(prediction)] * 100

    print("\n🤖 AEROCASTAI Prediction:")
    if prediction == 1:
        print(f"⚠️ Tornado Conditions Likely — Confidence: {confidence:.0f}%")
    else:
        print(f"✅ No Tornado Conditions Detected — Confidence: {confidence:.0f}%")

    log_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "slat": latitude,
        "slon": longitude,
        "len": wind_speed,
        "wid": wind_gusts,
        "temperature": temperature,
        "prediction": int(prediction),
        "confidence": round(confidence, 2),
        "location": location_name
    }

    log_df = pd.DataFrame([log_data])
    try:
        log_df.to_csv("daily_log.csv", mode="a", header=not pd.io.common.file_exists("daily_log.csv"), index=False)
    except Exception as e:
        print(f"⚠️ Failed to save log: {e}")

# ----------------------------
# Schedule to run daily at 7:00 AM, 3:00 PM, and 7:00 PM
# ----------------------------
schedule.every().day.at("07:00").do(run_aerocastai)
schedule.every().day.at("15:00").do(run_aerocastai)
schedule.every().day.at("19:00").do(run_aerocastai)
print("📅 Scheduler is active. AEROCASTAI will run at 7:00 AM, 3:00 PM, and 7:00 PM daily.")

while True:
    schedule.run_pending()
    time.sleep(60)
