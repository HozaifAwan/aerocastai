import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("daily_log.csv")

# Use correct feature columns
X = df[["slat", "slon", "len", "wid", "temperature"]]
y = df["prediction"]

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "aerocastai_model.pkl")
print("✅ Model saved as aerocastai_model.pkl")
