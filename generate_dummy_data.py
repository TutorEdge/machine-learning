# generate_dummy_data.py
import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)
np.random.seed(42)

# -----------------------------
# Create Tutor Data
# -----------------------------
subjects = ['Math', 'Science', 'English', 'History']
tutors = []
for i in range(1, 11):  # 10 tutors
    tutors.append({
        "tutor_id": f"T{i:03d}",
        "tutor_name": f"Tutor_{i}",
        "experience": np.random.randint(1, 15),
        "tutor_rating": round(np.random.uniform(3.0, 5.0), 2),  # scale 3–5
        "subject": np.random.choice(subjects)
    })

tutors_df = pd.DataFrame(tutors)
tutors_df.to_csv("data/tutors.csv", index=False)
print("✅ Created tutors.csv")

# -----------------------------
# Create Student Data
# -----------------------------
N = 200
students = []

for i in range(1, N + 1):
    name = f"Student_{i}"
    student_class = np.random.choice(['9A', '9B', '10A', '10B', '11A', '11B'])
    tutor = tutors_df.sample(1).iloc[0]
    tutor_id = tutor['tutor_id']

    # simulate marks progression
    test1 = np.random.randint(30, 95)
    test2 = int(np.clip(test1 + np.random.normal(0, 5), 0, 100))
    test3 = int(np.clip(test2 + np.random.normal(0, 5), 0, 100))

    students.append({
        "student_id": f"S{i:04d}",
        "name": name,
        "tutor_id": tutor_id,
        "class": student_class,
        "marks_test1": test1,
        "marks_test2": test2,
        "marks_test3": test3
    })

students_df = pd.DataFrame(students)
students_df.to_csv("data/students.csv", index=False)
print("✅ Created students.csv")
print(students_df.head())
