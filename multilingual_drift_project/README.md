cat > README.md <<EOF
# Topic Drift Detection

This project streams live news articles in English, Hindi, and Tamil from NewsAPI, detects latent topics using LDA, and visualizes topic drift using a Streamlit dashboard.

---

## 🛠 Requirements

- Python 3.10.x
- MongoDB (local)
- NewsAPI Key (get one from https://newsapi.org)

---

## 🔧 Setup Instructions

```bash
# 1. Create virtual environment
python3 -m venv drift_env310
source drift_env310/bin/activate

# 2. Install required libraries
pip install -r requirements.txt

# 3. Start MongoDB Server
mongod --dbpath ~/mongodb-data/db

# 4. Start Real-Time Ingestion
python scripts/real_time_ingestion.py

# 5. Start Dashboard
streamlit run dashboard/streamlit_app.py
