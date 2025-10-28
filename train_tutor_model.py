import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

# Load processed tutor recommendation data
df = pd.read_csv("data/processed_tutor.csv")

# Features & target
X = df[["avg_marks", "improvement_pct", "experience", "tutor_rating"]]
y = df["best_tutor_id"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest classifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Evaluate (optional)
acc = model.score(X_test, y_test)
print(f"✅ Tutor model trained! Test Accuracy: {acc:.2f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/tutor_model.pkl")
print("✅ Tutor model saved at models/tutor_model.pkl")
