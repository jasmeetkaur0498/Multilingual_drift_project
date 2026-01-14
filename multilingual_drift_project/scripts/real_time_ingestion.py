import time
from pymongo import MongoClient
import requests
import re
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK stopwords (run once if not already downloaded)
nltk.download('stopwords')
nltk.download('punkt')

# MongoDB Connection
client = MongoClient('mongodb://localhost:27017/')
db = client["topic_drift"]
collection = db["documents"]
topic_collection = db["topics"]

# NewsAPI Key
API_KEY = "187c99e3856e46c78f6530e65179ad25"

# English stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase and remove punctuation/numbers
    cleaned_text = re.sub(r'[^a-z\s]', ' ', text.lower())
    # Tokenize
    tokens = word_tokenize(cleaned_text)
    # Remove stopwords and short tokens
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    return tokens

def fetch_news():
    url = f"https://newsapi.org/v2/top-headlines?language=en&apiKey={API_KEY}"
    response = requests.get(url).json()
    articles = response.get("articles", [])
    for article in articles:
        text = article.get("title", "")
        if text:
            tokens = preprocess_text(text)
            collection.insert_one({
                "text": text,
                "language": "en",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "cleaned_tokens": tokens
            })

def train_lda():
    # Fetch all documents
    docs = list(collection.find({"language": "en"}))
    if not docs:
        return

    # Prepare corpus
    texts = [doc["cleaned_tokens"] for doc in docs if "cleaned_tokens" in doc]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Train LDA
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10, random_state=42)
    topics = lda_model.show_topics(num_topics=5, num_words=5, formatted=False)

    # Store topics
    topic_collection.drop()
    for topic_id, topic_words in topics:
        keywords = [word for word, _ in topic_words]
        topic_collection.insert_one({"topic_id": topic_id, "keywords": keywords})

# Main loop
last_lda_time = time.time()
while True:
    fetch_news()
    time.sleep(60)  # Fetch news every 60 seconds

    # Retrain LDA every 30 minutes
    if time.time() - last_lda_time >= 1800:
        train_lda()
        last_lda_time = time.time()





