import pandas as pd

# Load the existing daily_log.csv
df = pd.read_csv("daily_log.csv")

# Add missing columns if they don't already exist
new_columns = [
    "dew", "humidity", "precipitation", "cloudcover", "pressure", "cape", "lifted_index"
]

for col in new_columns:
    if col not in df.columns:
        df[col] = None

# Reorder columns to match the new model input
expected_order = [
    "timestamp", "slat", "slon", "len", "wid", "temperature",
    "dew", "humidity", "precipitation", "cloudcover", "pressure", "cape", "lifted_index",
    "prediction", "confidence", "location"
]

# Fill missing columns with None
for col in expected_order:
    if col not in df.columns:
        df[col] = None

# Reorder columns properly
df = df[expected_order]

# Save it back
df.to_csv("daily_log.csv", index=False)
print("✅ daily_log.csv has been updated with all required features.")
