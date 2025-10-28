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

# Calculate average marks (optional feature)
data["avg_marks"] = (data["marks_test1"] + data["marks_test2"] + data["marks_test3"]) / 3

# For demo: define best_tutor_id as the tutor of the student (can be replaced later)
# In real data, this can be based on historical improvement
data["best_tutor_id"] = data["tutor_id"]

# Select features for ML
features = ["avg_marks", "improvement_pct", "experience", "tutor_rating"]
target = "best_tutor_id"

X = data[features]
y = data[target]

# Scale numeric features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save processed data
processed = pd.DataFrame(X_scaled, columns=features)
processed["best_tutor_id"] = y
processed.to_csv("data/processed_tutor.csv", index=False)

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Save scaler
joblib.dump(scaler, "models/tutor_scaler.pkl")

print("✅ Tutor preprocessing complete! Saved processed_tutor.csv and scaler.")
