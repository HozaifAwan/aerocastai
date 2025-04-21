import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load and clean data
df = pd.read_csv("daily_log.csv").dropna(subset=["slat", "slon", "len", "wid", "temperature", "prediction"])

# Double-check if data is valid after dropping NaNs
if df.empty:
    raise ValueError("❌ Training dataset empty after dropping NaNs. Cannot retrain.")

# Use correct columns
X = df[["slat", "slon", "len", "wid", "temperature"]]
y = df["prediction"].astype(int)  # Ensure predictions are integers

# Retrain model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save retrained model
joblib.dump(model, "aerocastai_model.pkl")
print("✅ Model retrained and saved successfully")
