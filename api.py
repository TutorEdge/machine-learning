from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("models/performance_model.pkl")
scaler = joblib.load("models/scaler.pkl")

app = FastAPI(title="Performance Improvement Predictor API")

# Request schema
class StudentData(BaseModel):
    student_id: int
    marks_test1: float
    marks_test2: float
    experience: float
    tutor_rating: float

@app.post("/predict")
def predict_performance(data: StudentData):
    try:
        # Prepare input
        X = np.array([[data.marks_test1, data.marks_test2, data.experience, data.tutor_rating]])
        X_scaled = scaler.transform(X)
        # Predict percentage improvement
        predicted = model.predict(X_scaled)[0]

        return {
            "student_id": data.student_id,
            "predicted_improvement_pct": round(float(predicted), 2)
        }

    except Exception as e:
        return {"error": str(e)}
