import pandas as pd
import numpy as np
import random
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier

# ====== CONFIG ======
CSV_FILE = "daily_log.csv"
HEADER = [
    "timestamp", "lat", "lon", "temperature", "dew_point_2m", "relative_humidity_2m",
    "precipitation", "cloudcover", "surface_pressure", "convective_available_potential_energy",
    "lifted_index", "wind_speed_10m", "wind_gusts_10m", "prediction", "confidence", "location"
]

# ====== CLEAN CSV LOG FUNCTION ======
def clean_csv():
    try:
        df = pd.read_csv(CSV_FILE, header=0, skip_blank_lines=True, dtype=str)
        df = df[df.apply(lambda row: len(row.dropna()) == len(HEADER), axis=1)]
        df.to_csv(CSV_FILE, index=False)
        print("🧹 Cleaned and fixed daily_log.csv")
    except Exception as e:
        print(f"⚠️ Failed to clean CSV: {e}")

# ====== MAIN LOGIC ======
def run():
    clean_csv()

    try:
        print("🔁 Retraining model...")
        df = pd.read_csv(CSV_FILE)
        df = df.dropna()
        df = df[df['prediction'].isin([0, 1])]

        if df.empty:
            print("❌ Not enough valid rows to retrain the model.")
            return

        expected_cols = [
            "lat", "lon", "wind_speed_10m", "wind_gusts_10m", "temperature",
            "dew_point_2m", "relative_humidity_2m", "precipitation", "cloudcover",
            "surface_pressure", "convective_available_potential_energy", "lifted_index"
        ]

        for col in expected_cols:
            if col not in df.columns:
                print(f"⚠️ Missing column in CSV: {col}")
                return

        X = df[expected_cols].astype(float)
        y = df["prediction"].astype(int)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        joblib.dump(model, "aerocastai_model.pkl")
        print("✅ Model retrained and saved as aerocastai_model.pkl")
    except Exception as e:
        print(f"❌ Model retraining failed: {e}")

if __name__ == "__main__":
    run()
