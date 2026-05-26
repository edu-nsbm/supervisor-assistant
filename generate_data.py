import pandas as pd
import numpy as np
from faker import Faker
import random

fake = Faker()

NUM_STUDENTS = 250
BATCH = "26.1"

data = []

for i in range(NUM_STUDENTS):
    student_id = f"STU-{i+1:03d}"
    name = fake.name()
    
    prog_fundamentals = np.clip(np.random.normal(65, 15), 0, 100)
    math_computing = np.clip(np.random.normal(60, 20), 0, 100)
    intro_cs = np.clip(np.random.normal(70, 12), 0, 100)
    db_management = np.clip(np.random.normal(68, 14), 0, 100)
    personal_dev = np.clip(np.random.normal(75, 10), 0, 100)
    
    attendance = np.clip(np.random.normal(0.8, 0.15), 0.0, 1.0)
    late_submissions = max(0, int(np.random.normal(2, 3)))

    risk_score = 0.0
    
    if prog_fundamentals < 50: risk_score += 0.4
    if math_computing < 50: risk_score += 0.3
    if attendance < 0.75: risk_score += 0.3
    if late_submissions > 3: risk_score += 0.2

    risk_score += random.uniform(-0.1, 0.1)
    
    target_failed_ai = 1 if risk_score > 0.4 else 0

    data.append({
        "student_id": student_id,
        "name": name,
        "batch": BATCH,
        "Programming Fundamentals": round(prog_fundamentals),
        "Mathematics for Computing": round(math_computing),
        "Introduction to Computer Science": round(intro_cs),
        "Database Management Systems": round(db_management),
        "Personal Development": round(personal_dev),
        "attendance_percentage": round(attendance, 2),
        "late_submissions_count": late_submissions,
        "Target_Failed_AI": target_failed_ai
    })

df = pd.DataFrame(data)
df.to_csv("student_records_batch26_1.csv", index=False)
print(f"Generated {NUM_STUDENTS} records successfully!")