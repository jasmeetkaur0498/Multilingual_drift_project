from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# === 1. Connect to MongoDB ===
client = MongoClient('mongodb://localhost:27017/')
db = client["topic_drift"]
topic_collection = db["topics"]
drift_collection = db["topic_drift_scores"]

# === 2. Fetch Topics ===
topics = list(topic_collection.find())

# Prepare topic texts (concatenate keywords)
topic_texts = [" ".join(topic["keywords"]) for topic in topics]
topic_ids = [topic["topic_id"] for topic in topics]

# === 3. Calculate TF-IDF Vectors ===
vectorizer = TfidfVectorizer()
topic_vectors = vectorizer.fit_transform(topic_texts)

# === 4. Compare Topics Pairwise Using Cosine Similarity ===
similarity_matrix = cosine_similarity(topic_vectors)

# === 5. Detect Drift Between Topics and Save with Timestamp ===
drift_threshold = 0.7  # Similarity < 0.7 = drift detected

for i in range(len(topic_ids)):
    for j in range(i + 1, len(topic_ids)):
        similarity_score = similarity_matrix[i, j]
        drift_detected = similarity_score < drift_threshold

        drift_doc = {
            "topic_id_1": int(topic_ids[i]),
            "topic_id_2": int(topic_ids[j]),
            "similarity_score": round(float(similarity_score), 4),
            "drift_detected": bool(drift_detected),
            "timestamp": datetime.now()  # ✅ Real timestamp added here
        }

        drift_collection.insert_one(drift_doc)

print("✅ Drift detection completed and stored into MongoDB with timestamps!")
