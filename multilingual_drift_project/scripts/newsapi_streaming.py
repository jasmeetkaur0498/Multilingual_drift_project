import requests
import time
from pymongo import MongoClient
from datetime import datetime

# === Your NewsAPI Key ===
API_KEY = "187c99e3856e46c78f6530e65179ad25"  # 👈 Replace here inside quotes

# === MongoDB Connection ===
mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client["topic_drift"]
collection = db["documents"]

# === Function to fetch news ===
def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?language=en&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    articles = data.get("articles", [])
    return articles

print("✅ News Streaming Started... Fetching every 30 seconds...")

# === Infinite Streaming Loop ===
while True:
    try:
        articles = fetch_news()
        for article in articles:
            text = (article.get('title') or '') + " " + (article.get('description') or '')
            if text.strip() == "":
                continue
            doc = {
                "text": text,
                "language": "en",
                "timestamp": datetime.now()
            }
            collection.insert_one(doc)
            print(f"✅ Inserted: {text[:60]}...")

    except Exception as e:
        print(f"❌ Error: {e}")

    # Wait for 30 seconds before fetching again
    time.sleep(30)
