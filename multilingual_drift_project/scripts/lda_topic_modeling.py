from pymongo import MongoClient
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Download NLTK resources if not already
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# MongoDB Connection
client = MongoClient('mongodb://localhost:27017/')
db = client["topic_drift"]
collection = db["documents"]

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# 🧠 Updated smart Preprocess
def preprocess(text, lang="en"):
    text = text.lower()
    if lang == "en":
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    else:
        text = re.sub(r'[0-9]', '', text)  # remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuations
        tokens = text.split()
    return tokens

# Fetch documents by language
languages = ['en', 'ta', 'hi']

for lang in languages:
    print("="*50)
    print(f"Training LDA Model for Language: {lang}")
    print("="*50)
    
    docs = []
    for doc in collection.find({"language": lang}):
        tokens = preprocess(doc['text'], lang=doc['language'])  # ✅ Pass language correctly
        if tokens:
            docs.append(tokens)
    
    if len(docs) < 2:
        print(f"Not enough documents for language: {lang}. Skipping.")
        continue

    # Create Dictionary and Corpus
    dictionary = Dictionary(docs)
    corpus = [dictionary.doc2bow(text) for text in docs]

    # Train LDA Model
    lda_model = LdaModel(corpus=corpus, num_topics=3, id2word=dictionary, passes=10, random_state=42)

    # Print Topics
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)
    print("\n\n")

