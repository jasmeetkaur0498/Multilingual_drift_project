from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('punkt')

client = MongoClient('mongodb://localhost:27017/')
db = client["topic_drift"]
collection = db["documents"]

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    cleaned_text = re.sub(r'[^a-z\s]', ' ', text.lower())
    tokens = word_tokenize(cleaned_text)
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    return tokens

docs = collection.find({"language": "en"})
for doc in docs:
    if "cleaned_tokens" not in doc:
        text = doc.get("text", "")
        tokens = preprocess_text(text)
        collection.update_one({"_id": doc["_id"]}, {"$set": {"cleaned_tokens": tokens}})