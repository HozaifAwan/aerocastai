import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the log CSV
df = pd.read_csv("daily_log.csv")

# Keep only rows with valid prediction and no missing values
df = df.dropna()
df = df[df['prediction'].isin([0, 1])]

# If there are no valid rows, quit
if df.empty:
    print("❌ Not enough valid rows to retrain the model. Run live_predictor.py manually first.")
    exit()

# Select features for training
features = [
    "slat", "slon", "len", "wid", "temperature",
    "dew", "humidity", "precipitation", "cloudcover",
    "pressure", "cape", "lifted_index"
]

X = df[features]
y = df["prediction"]

# Retrain the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the new model
joblib.dump(model, "aerocastai_model.pkl")
print("✅ Model retrained and saved as aerocastai_model.pkl")
