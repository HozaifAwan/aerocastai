import pandas as pd
import requests
import joblib
import os
import random
import base64
from datetime import datetime
from retrain_model import run as retrain_model

# CONFIG
repo_name = "HozaifAwan/aerocastai"
github_token = os.environ.get("GH_TOKEN")
csv_file = "daily_log.csv"
model_file = "aerocastai_model.pkl"
log_file = "retrain_log.txt"

# Fetch latest CSV from GitHub
def download_csv_from_github(repo, token):
    url = f"https://api.github.com/repos/{repo}/contents/{csv_file}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.raw"
    }
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write(r.text)
        print("⬇️ Pulled latest daily_log.csv from GitHub")
    else:
        print("⚠️ No previous CSV found or failed to pull — will create fresh one")

# Upload file to GitHub
def upload_to_github(filepath, message, repo, token):
    filename = os.path.basename(filepath)
    url = f"https://api.github.com/repos/{repo}/contents/{filename}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }

    # Get SHA of existing file
    sha = None
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        sha = r.json().get("sha")

    # Read and encode file
    with open(filepath, "rb") as f:
        content = base64.b64encode(f.read()).decode()

    data = {
        "message": message,
        "content": content,
    }
    if sha:
        data["sha"] = sha

    r = requests.put(url, headers=headers, json=data)
    if r.status_code in [200, 201]:
        print(f"✅ Uploaded {filename} to GitHub")
    else:
        print(f"❌ Failed to upload {filename}: {r.status_code} — {r.text}")

# Start fresh if GitHub token available
if github_token:
    download_csv_from_github(repo_name, github_token)

# Step 1: Random location
lat = round(random.uniform(25, 49), 4)
lon = round(random.uniform(-124, -66), 4)

# Step 2: Get Open-Meteo weather data
url = (
    f"https://api.open-meteo.com/v1/forecast"
    f"?latitude={lat}&longitude={lon}"
    f"&hourly=wind_speed_10m,wind_gusts_10m,temperature_2m,dew_point_2m,"
    f"relative_humidity_2m,precipitation,cloudcover,surface_pressure,"
    f"cape,lifted_index"
    f"&forecast_days=1&timezone=America%2FChicago"
)
response = requests.get(url).json()
latest = 0

# Step 3: Extract values
def get(field):
    return response["hourly"].get(field, [None])[latest]

data = {
    "lat": lat,
    "lon": lon,
    "wind_speed_10m": get("wind_speed_10m"),
    "wind_gusts_10m": get("wind_gusts_10m"),
    "temperature": get("temperature_2m"),
    "dew_point_2m": get("dew_point_2m"),
    "relative_humidity_2m": get("relative_humidity_2m"),
    "precipitation": get("precipitation"),
    "cloudcover": get("cloudcover"),
    "surface_pressure": get("surface_pressure"),
    "convective_available_potential_energy": get("cape"),
    "lifted_index": get("lifted_index")
}

# Step 4: Predict
model = joblib.load(model_file)
X_input = pd.DataFrame([data])
prediction = int(model.predict(X_input)[0])
probas = model.predict_proba(X_input)[0]
confidence = round(probas[prediction] * 100, 2)

# Step 5: Log row
log_entry = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    **data,
    "prediction": prediction,
    "confidence": confidence,
    "location": "Random US coords"
}
log_df = pd.DataFrame([log_entry])

# Step 6: Append to CSV
log_df.to_csv(csv_file, mode="a", header=not os.path.exists(csv_file), index=False)
print(f"📥 Logged random data entry at {log_entry['timestamp']}")

# Step 7: Retrain model
retrain_model()

# Step 8: Upload files
if github_token:
    upload_to_github(csv_file, f"Update log {log_entry['timestamp']}", repo_name, github_token)
    upload_to_github(model_file, f"Update model {log_entry['timestamp']}", repo_name, github_token)
    if os.path.exists(log_file):
        upload_to_github(log_file, f"Update retrain log {log_entry['timestamp']}", repo_name, github_token)
else:
    print("⚠️ GH_TOKEN not found in environment.")
