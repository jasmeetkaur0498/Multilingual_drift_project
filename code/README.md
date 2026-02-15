# 🧠 Multilingual Topic Drift Detection System

A comprehensive Python-based system that detects, analyzes, and visualizes topic drift in real-time news articles across multiple languages (English, Hindi, Tamil). The system streams live news from NewsAPI, performs advanced NLP preprocessing, applies Latent Dirichlet Allocation (LDA) for topic modeling, detects semantic drift using cosine similarity, and provides interactive visualizations through a Streamlit dashboard.

**Key Features:**
- 📡 Real-time news ingestion from NewsAPI
- 🌍 Multilingual support (English, Hindi, Tamil)
- 🧮 LDA-based topic modeling with 3-5 topics per language
- 📊 Drift detection using TF-IDF and cosine similarity
- 📈 Interactive Streamlit dashboard with visualizations
- 💾 MongoDB for persistent data storage
- 🐳 Docker containerization support
- 🤖 PySpark machine learning predictions for drift classification

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [System Components](#system-components)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Flow](#data-flow)
- [APIs & Endpoints](#apis--endpoints)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

---

## 🏗️ Architecture Overview

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      TOPIC DRIFT DETECTION SYSTEM               │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────┐
│   NewsAPI Server     │  (External Data Source)
│  - English Articles  │
└──────────┬───────────┘
           │ HTTP GET Request (every 30 sec)
           ▼
┌──────────────────────────────────────────┐
│   Real-Time News Ingestion Module        │
│   (newsapi_streaming.py)                 │
│  - Fetch articles every 30 seconds      │
│  - Extract title + description          │
│  - Add timestamp metadata               │
└──────────┬───────────────────────────────┘
           │ Insert Documents
           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      MongoDB Database                             │
│  ┌─────────────────────┬─────────────────────┬─────────────────┐ │
│  │ documents           │ topics              │ topic_drift_    │ │
│  │ - text              │ - topic_id          │  scores         │ │
│  │ - language          │ - keywords[]        │ - topic_id_1    │ │
│  │ - timestamp         │                     │ - topic_id_2    │ │
│  │ - cleaned_tokens[]  │                     │ - similarity    │ │
│  │                     │                     │ - drift_        │ │
│  │                     │                     │   detected      │ │
│  └─────────────────────┴─────────────────────┴─────────────────┘ │
└─────┬──────────────────────────────────────────────┬──────────────┘
      │ Read & Process                              │ Query Results
      ▼                                              ▼
┌──────────────────────────────────────────┐  ┌──────────────────────┐
│  Preprocessing Pipeline                  │  │   Drift Detection    │
│  (lda_topic_modeling.py)                 │  │  (detect_topic_drift)│
│                                          │  │                      │
│  • Language detection                    │  │  • TF-IDF Vectors   │
│  • Tokenization & normalization         │  │  • Cosine Similarity│
│  • Remove stopwords & punctuation       │  │  • Threshold: 0.7   │
│  • Lemmatization                        │  │  • Timestamp logs   │
│  • Create Dictionary & Corpus           │  │                      │
└──────────────────┬───────────────────────┘  └──────────┬───────────┘
                   │                                     │
                   ▼                                     ▼
         ┌──────────────────────┐          ┌──────────────────────────┐
         │  LDA Model Training  │          │  Drift Score Calculation │
         │  (Gensim)            │          │                          │
         │  • 3-5 topics        │          │  • Compare topic vectors │
         │  • 10 passes         │          │  • Store results to DB   │
         │  • Per language      │          │  • Track over time       │
         └──────────┬───────────┘          └──────────┬───────────────┘
                    │                                 │
                    └─────────────────┬────────────────┘
                                      │
                    ┌─────────────────▼────────────────┐
                    │  PySpark ML Pipeline             │
                    │ (predict_topic_drift.py)         │
                    │  • Load from MongoDB             │
                    │  • Feature engineering           │
                    │  • Logistic Regression model     │
                    │  • Performance metrics (ROC, F1) │
                    └──────────────────┬───────────────┘
                                       │
                    ┌──────────────────▼───────────────┐
                    │   Streamlit Dashboard            │
                    │  (dashboard/streamlit_app.py)    │
                    │  📊 Document browser             │
                    │  📊 Topic keywords               │
                    │  ☁️  WordClouds                  │
                    │  📈 Similarity charts            │
                    │  📉 Drift trends over time       │
                    └──────────────────────────────────┘
```

---

## 🔧 System Components

### 1. **Data Ingestion Layer**
- **newsapi_streaming.py** - Fetches live news articles every 30 seconds
- **real_time_ingestion.py** - Background ingestion with preprocessing

### 2. **Preprocessing & NLP Layer**
- **preprocess_text.py** - Text cleaning and tokenization
- **lda_topic_modeling.py** - Multilingual preprocessing with lemmatization

### 3. **Topic Modeling Layer**
- **Gensim LDA Model** - Generates latent topics per language
- Configurable topics (3-5) with 10 training passes
- Per-language topic models (English, Hindi, Tamil)

### 4. **Drift Detection Layer**
- **detect_topic_drift.py** - Calculates cosine similarity between topics
- TF-IDF vectorization for semantic representation
- Drift threshold: 0.7 (configurable)
- Timestamp logging for trend analysis

### 5. **ML Prediction Layer**
- **predict_topic_drift.py** - PySpark-based drift prediction
- Logistic Regression classification
- Performance metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 6. **Visualization Layer**
- **Streamlit Dashboard** - Interactive real-time analytics
- Multi-language document browser
- WordCloud visualizations
- Time-series drift trends

---

## 💻 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Language** | Python | 3.10.x |
| **Data Collection** | NewsAPI, Requests | - |
| **NLP Framework** | SpaCy, NLTK | 3.8.5, 3.9.1 |
| **Topic Modeling** | Gensim | 4.3.3 |
| **ML Framework** | PySpark, scikit-learn | 3.3.2, 1.6.1 |
| **Database** | MongoDB | Latest |
| **Visualization** | Streamlit, Plotly | 1.44.1, 6.0.1 |
| **Infrastructure** | Docker | Latest |

---

## 📦 Prerequisites

- **Python** 3.10 or higher
- **MongoDB** (local or containerized via Docker)
- **NewsAPI** API Key (https://newsapi.org)
- **macOS/Linux/Windows** with terminal access

### System Requirements
- Minimum 4GB RAM
- 2GB storage for MongoDB data
- Stable internet connection

---

## 🚀 Installation

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd multilingual_drift_project
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv drift_env310
source drift_env310/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```bash
python -c "
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
"
```

### Step 5: Setup MongoDB

**Option A: Local Installation**
```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

**Option B: Docker**
```bash
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

---

## ⚙️ Configuration

### NewsAPI Configuration
1. Sign up at https://newsapi.org
2. Update API key in:
   - `scripts/newsapi_streaming.py` (line 6)
   - `scripts/real_time_ingestion.py` (line 20)

```python
API_KEY = "YOUR_NEWSAPI_KEY_HERE"
```

### LDA Model Parameters
In `scripts/lda_topic_modeling.py`:
```python
num_topics = 3          # Number of topics (3-5 recommended)
passes = 10             # Training passes
```

### Drift Detection Threshold
In `scripts/detect_topic_drift.py`:
```python
drift_threshold = 0.7   # Similarity < 0.7 = drift detected
```

---

## 📖 Usage

### Complete Pipeline (Recommended)

**Terminal 1: Start MongoDB**
```bash
mongod --dbpath ~/mongodb-data/db
```

**Terminal 2: Start News Ingestion**
```bash
source drift_env310/bin/activate
python scripts/newsapi_streaming.py
```

**Terminal 3: Run Preprocessing (after 2-3 minutes)**
```bash
source drift_env310/bin/activate
python scripts/preprocess_text.py
```

**Terminal 4: Train LDA Model**
```bash
source drift_env310/bin/activate
python scripts/lda_topic_modeling.py
```

**Terminal 5: Detect Topic Drift**
```bash
source drift_env310/bin/activate
python scripts/detect_topic_drift.py
```

**Terminal 6: Launch Dashboard**
```bash
source drift_env310/bin/activate
streamlit run dashboard/streamlit_app.py
```

Access dashboard at `http://localhost:8501`

---

## 📁 Project Structure

```
multilingual_drift_project/
├── README.md                          # Project documentation
├── requirements.txt                   # Dependencies
├── scripts/                           # Core processing modules
│   ├── newsapi_streaming.py          # Real-time ingestion
│   ├── preprocess_text.py            # Text preprocessing
│   ├── lda_topic_modeling.py         # Topic modeling
│   ├── detect_topic_drift.py         # Drift detection
│   └── predict_topic_drift.py        # ML predictions
└── dashboard/                         # Visualization
    └── streamlit_app.py              # Interactive dashboard
```

---

## 🔄 Data Flow Diagram

```
NewsAPI
   ↓
newsapi_streaming.py
   ↓ (Insert documents)
MongoDB: documents collection
   ↓ (Read articles)
preprocess_text.py
   ↓ (Update cleaned_tokens)
MongoDB: documents (updated)
   ↓ (Read preprocessed docs)
lda_topic_modeling.py
   ↓ (Extract topics per language)
MongoDB: topics collection
   ↓ (Read topics)
detect_topic_drift.py
   ↓ (Calculate similarity)
MongoDB: topic_drift_scores collection
   ↓ (Query all collections)
Streamlit Dashboard
   ↓
Interactive Visualizations (http://localhost:8501)
```

---

## 🔌 MongoDB Collections Schema

### documents Collection
```json
{
  "_id": ObjectId,
  "text": "Article text",
  "language": "en|hi|ta",
  "timestamp": ISODate,
  "cleaned_tokens": ["token1", "token2"]
}
```

### topics Collection
```json
{
  "_id": ObjectId,
  "topic_id": 0,
  "keywords": ["word1", "word2", "word3"]
}
```

### topic_drift_scores Collection
```json
{
  "_id": ObjectId,
  "topic_id_1": 0,
  "topic_id_2": 1,
  "similarity_score": 0.75,
  "drift_detected": true,
  "timestamp": ISODate
}
```

---

## 📊 Performance Metrics

| Metric | Expected Value |
|--------|---|
| News Ingestion Rate | 20 articles/30 sec |
| Preprocessing Time | 100 ms per document |
| LDA Training Time | 10-30 seconds (100+ docs) |
| Drift Detection Time | 50 ms per topic pair |
| Dashboard Load Time | <2 seconds |

---

## 🐛 Troubleshooting

### MongoDB Connection Error
```bash
# Check if running
ps aux | grep mongod

# Start MongoDB
mongod --dbpath ~/mongodb-data/db

# Or Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### NewsAPI Rate Limit
- Free tier: 100 requests/day
- Upgrade at https://newsapi.org/pricing
- Increase sleep interval in `newsapi_streaming.py`

### No Topics Generated
1. Run ingestion longer (5-10 minutes)
2. Check MongoDB for documents:
```bash
python -c "from pymongo import MongoClient; print(MongoClient()['topic_drift']['documents'].count_documents({}))"
```

### Streamlit Issues
```bash
streamlit cache clear
streamlit run dashboard/streamlit_app.py --logger.level=debug
```

---

## 🚀 Future Enhancements

1. **Real-time Alerts** - Email/Slack notifications
2. **Sentiment Analysis** - Alongside topic drift
3. **Better Models** - Top2Vec, BERTopic
4. **Scalability** - Kafka, Kubernetes
5. **UI Improvements** - WebSockets, exports
6. **More Languages** - Spanish, French, German
7. **MLOps** - Model versioning, A/B testing

---

## 📝 Example Scenarios

### Scenario 1: News Trend Analysis
1. Run ingestion for 1 hour
2. Generate 5 topics
3. Observe similarity trends
4. Identify emerging vs. persistent topics

### Scenario 2: Multilingual Comparison
1. Collect news in English, Hindi, Tamil
2. Train separate models per language
3. Cross-reference keyword overlap
4. Identify global vs. localized topics

### Scenario 3: Anomaly Detection
1. Establish baseline drift scores
2. Monitor in real-time
3. Flag anomalies when similarity < threshold
4. Investigate root causes

---

## 🔗 References

- [Gensim LDA](https://radimrehurek.com/gensim/models/ldamodel.html)
- [NewsAPI](https://newsapi.org/docs)
- [MongoDB Python](https://pymongo.readthedocs.io/)
- [Streamlit](https://docs.streamlit.io/)
- [PySpark MLlib](https://spark.apache.org/docs/latest/ml-guide.html)
- [NLTK](https://www.nltk.org/)
- [SpaCy](https://spacy.io/)
