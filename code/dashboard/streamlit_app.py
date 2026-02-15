import streamlit as st
import pandas as pd
import plotly.express as px
from pymongo import MongoClient
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# === MongoDB Connection ===
client = MongoClient('mongodb://localhost:27017/')
db = client["topic_drift"]
topic_collection = db["topics"]
drift_collection = db["topic_drift_scores"]
document_collection = db["documents"]

# === Streamlit Settings ===
st.set_page_config(page_title="Multilingual Topic Drift Detection", layout="wide")
st.title("🧠 Multilingual Topic Drift Detection Dashboard")

# === 1. Language Filter for Documents ===
available_languages = list({doc.get("language", "unknown") for doc in document_collection.find()})
selected_language = st.selectbox("Select Language to Filter Documents", options=available_languages)

filtered_docs = list(document_collection.find({"language": selected_language}))

st.subheader(f"Documents in Language: {selected_language}")

if filtered_docs:
    doc_df = pd.DataFrame([{
        "Text": doc.get("text", ""),
        "Tokens": ", ".join(doc.get("cleaned_tokens", []))
    } for doc in filtered_docs])
    st.dataframe(doc_df, use_container_width=True)
else:
    st.warning(f"No documents found for language: {selected_language}")

# === 2. Discovered Topics ===
st.subheader("📝 Discovered Topics")

topics = list(topic_collection.find())
topics_df = pd.DataFrame([{
    "Topic ID": topic["topic_id"],
    "Keywords": ", ".join(topic["keywords"])
} for topic in topics])

if not topics_df.empty:
    st.dataframe(topics_df, use_container_width=True)
else:
    st.warning("No topics discovered yet.")

# === 3. WordClouds for Topics ===
st.subheader("☁️ WordClouds for Topics")

for topic in topics:
    keywords = topic.get("keywords", [])
    if keywords:
        text = " ".join(keywords)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

        st.markdown(f"### Topic {topic['topic_id']}")
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# === 4. Drift Similarity Bar Chart ===
st.subheader("📊 Topic Drift Similarity Bar Chart")

drift_scores = list(drift_collection.find())

if drift_scores:
    drift_df = pd.DataFrame(drift_scores)

    if not drift_df.empty:
        drift_df["Drift Detected"] = drift_df["drift_detected"].apply(lambda x: "Yes" if x else "No")

        fig = px.bar(
            drift_df,
            x="topic_id_1",
            y="similarity_score",
            color="Drift Detected",
            barmode="group",
            title="Similarity Score between Topics",
            labels={"similarity_score": "Cosine Similarity", "topic_id_1": "Topic ID 1"}
        )

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(drift_df[["topic_id_1", "topic_id_2", "similarity_score", "Drift Detected"]], use_container_width=True)
    else:
        st.warning("No drift data available.")
else:
    st.warning("No drift scores found.")

# === 5. Topic Drift Trend Over Time ===
st.subheader("📈 Topic Drift Trend Over Time")

if drift_scores:
    drift_trend_df = pd.DataFrame(drift_scores)

    if not drift_trend_df.empty and 'timestamp' in drift_trend_df.columns:
        drift_trend_df['timestamp'] = pd.to_datetime(drift_trend_df['timestamp'])

        topic_pairs = drift_trend_df[['topic_id_1', 'topic_id_2']].drop_duplicates()
        topic_pair_selection = st.selectbox(
            "Select Topic Pair to View Trend",
            options=[(row['topic_id_1'], row['topic_id_2']) for _, row in topic_pairs.iterrows()]
        )

        selected_trend = drift_trend_df[
            (drift_trend_df['topic_id_1'] == topic_pair_selection[0]) &
            (drift_trend_df['topic_id_2'] == topic_pair_selection[1])
        ]

        if not selected_trend.empty:
            fig = px.line(
                selected_trend,
                x="timestamp",
                y="similarity_score",
                title=f"Drift Trend Between Topic {topic_pair_selection[0]} and Topic {topic_pair_selection[1]}",
                labels={"similarity_score": "Similarity", "timestamp": "Time"}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No trend data for selected topic pair.")
    else:
        st.warning("No timestamp data available.")





