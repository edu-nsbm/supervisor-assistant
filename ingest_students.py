import pandas as pd
import joblib
import chromadb
from chromadb.utils import embedding_functions

print("Loading data and ML model...")
df = pd.read_csv("student_records_batch26_1.csv")
model = joblib.load("ai_risk_predictor.pkl")

chroma_client = chromadb.PersistentClient(path="./data/vectordb")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection(
    name="student_profiles",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

ids, docs, metas = [], [], []

print("Processing student profiles...")
for index, row in df.iterrows():
    student_id = row['student_id']
    
    features = pd.DataFrame([row.drop(["student_id", "name", "batch", "Target_Failed_AI"])])
    
    risk_probability = model.predict_proba(features)[0][1] * 100 
    
    profile_text = f"""
    Student Name: {row['name']} (ID: {student_id}, Batch: {row['batch']})
    ---
    Academic Performance:
    - Programming Fundamentals: {row['Programming Fundamentals']}/100
    - Introduction to Computer Science: {row['Introduction to Computer Science']}/100
    - Mathematics for Computing: {row['Mathematics for Computing']}/100
    - Database Management Systems: {row['Database Management Systems']}/100
    - Personal Development: {row['Personal Development']}/100
    
    Behavioral Metrics:
    - Attendance: {row['attendance_percentage'] * 100}%
    - Late Submissions: {row['late_submissions_count']}
    
    AI System Alert:
    The predictive model has calculated a {risk_probability:.1f}% risk of this student failing the upcoming Fundamentals of AI module based on their current academic trajectory.
    """
    
    ids.append(student_id)
    docs.append(profile_text.strip())
    metas.append({
        "student_id": student_id,
        "batch": str(row['batch']),
        "risk_percentage": float(risk_probability),
        "needs_intervention": bool(risk_probability > 50.0)
    })

print("Embedding and storing in ChromaDB... (This might take a minute)")
collection.add(ids=ids, documents=docs, metadatas=metas)
print(f"Successfully ingested {len(ids)} student profiles into the vector database!")