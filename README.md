
# TutorEdge Machine Learning APIs

Simple ML demo project using FastAPI.
This API predicts student improvement percentage based on input features.

---

## 1) Clone the project

```bash
git clone https://github.com/TutorEdge/machine-learning.git
cd machine-learning
````

---

## 2) Create Virtual Environment (Windows)

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3) Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4) Run API server

```bash
uvicorn api:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

Swagger UI (API testing UI):

```
http://127.0.0.1:8000/docs
```

---

## 5) Test Prediction API

### Endpoint

```
POST http://127.0.0.1:8000/predict
```

### Body Example

```json
{
  "student_id": 0,
  "marks_test1": 0,
  "marks_test2": 0,
  "experience": 0,
  "tutor_rating": 0
}
```

### Sample Response

```json
{
  "student_id": 0,
  "predicted_improvement_pct": 1.21
}
```

---

## Project Structure

| File           | Description                   |
| -------------- | ----------------------------- |
| api.py         | FastAPI app for predictions   |
| train_model.py | Script to train student model |
| preprocess.py  | Data preprocessing logic      |
| data/          | datasets                      |
| models/        | saved ML models               |

---

## Notes

* Always activate `venv` before running API
* Use `/docs` to test endpoints easily
