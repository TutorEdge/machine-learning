import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Load processed data
df = pd.read_csv("data/processed.csv")

# Features & target
X = df[["marks_test1", "marks_test2", "experience", "tutor_rating"]]
y = df["improvement_pct"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained!")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/performance_model.pkl")
print("✅ Model saved at models/performance_model.pkl")
