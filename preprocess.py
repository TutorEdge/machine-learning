import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

# Load CSVs
students = pd.read_csv("data/students.csv")
tutors = pd.read_csv("data/tutors.csv")

# Merge student + tutor info on tutor_id
data = pd.merge(students, tutors, on="tutor_id", how="left")

# Calculate percentage improvement (Test3 vs Test2)
data["improvement_pct"] = ((data["marks_test3"] - data["marks_test2"]) / data["marks_test2"]) * 100

# Select features
features = ["marks_test1", "marks_test2", "experience", "tutor_rating"]
target = "improvement_pct"

X = data[features]
y = data[target]

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save processed data
processed = pd.DataFrame(X_scaled, columns=features)
processed["improvement_pct"] = y
processed.to_csv("data/processed.csv", index=False)

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Data preprocessing complete! Saved processed.csv and scaler.pkl")
