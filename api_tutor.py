from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Tutor Recommendation API")

# Load trained model and scaler
model = joblib.load("models/tutor_model.pkl")
scaler = joblib.load("models/tutor_scaler.pkl")

# Input schema
class StudentFeatures(BaseModel):
    student_id: int
    avg_marks: float
    improvement_pct: float
    experience: float
    tutor_rating: float

@app.post("/recommend")
def recommend_tutor(data: StudentFeatures):
    try:
        X = np.array([[data.avg_marks, data.improvement_pct, data.experience, data.tutor_rating]])
        X_scaled = scaler.transform(X)
        tutor_id = model.predict(X_scaled)[0]

        return {
            "student_id": data.student_id,
            "recommended_tutor_id": str(tutor_id)  # Keep as string
        }

    except Exception as e:
        return {"error": str(e)}

