import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime

# Load the log data
try:
    data = pd.read_csv("daily_log.csv", usecols=["slat", "slon", "len", "wid", "temperature", "prediction", "confidence"])
    print("🧪 Columns in CSV:", list(data.columns))
except FileNotFoundError:
    print("❌ daily_log.csv not found. Make sure you have log data.")
    exit()

# Check if required columns exist
required_columns = {"slat", "slon", "len", "wid", "temperature", "prediction"}
if not required_columns.issubset(data.columns):
    print("❌ daily_log.csv is missing required columns.")
    exit()

# Define features and label
X = data[["slat", "slon", "len", "wid", "temperature"]]
y = data["prediction"]

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions) * 100
print(f"🎯 Model Accuracy on Test Data: {accuracy:.2f}%")

# Save model only if it's reasonably accurate
if accuracy >= 70:
    joblib.dump(model, "aerocastai_model.pkl")
    print("💾 Updated model saved as 'aerocastai_model.pkl'")
else:
    print("⚠️ Model not saved due to low accuracy.")
