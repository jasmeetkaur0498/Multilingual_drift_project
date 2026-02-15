from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split, udf, when
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.clustering import LDA

# === Create Spark Session ===
spark = SparkSession.builder \
    .appName("TopicDriftDetection") \
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/topic_drift.documents") \
    .config("spark.mongodb.write.connection.uri", "mongodb://127.0.0.1/topic_drift.output") \
    .config("spark.jars", \
            "/Users/jasmeetkaur/Downloads/mongo-spark-connector_2.12-10.2.1.jar," \
            "/Users/jasmeetkaur/Downloads/mongodb-driver-sync-4.11.1.jar," \
            "/Users/jasmeetkaur/Downloads/mongodb-driver-core-4.11.1.jar," \
            "/Users/jasmeetkaur/Downloads/bson-4.11.1.jar") \
    .getOrCreate()

# === Load Data ===
df = spark.read.format("mongodb").load()

# Show raw data
print("=== Raw Data ===")
df.select("_id", "language", "text", "timestamp").show(truncate=False)

# === Step 1: Preprocessing ===

# 1. Lowercase text
df = df.withColumn("text_clean", lower(col("text")))

# 2. Language-specific Cleaning (English clean heavy, Tamil/Hindi minimal)
df = df.withColumn(
    "text_clean",
    when(col("language") == "en", regexp_replace(col("text_clean"), r"[^a-zA-Z\s]", ""))
    .otherwise(col("text_clean"))
)

# 3. Tokenize (split words)
df = df.withColumn("tokens", split(col("text_clean"), " "))

# 4. Remove empty tokens
@udf(ArrayType(StringType()))
def remove_empty_tokens(words):
    return [word for word in words if word.strip() != ""]

df = df.withColumn("tokens", remove_empty_tokens(col("tokens")))

# 5. Remove Stopwords (only English)
remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
df = remover.transform(df)

df = df.withColumn(
    "filtered_tokens",
    when(col("language") == "en", col("filtered_tokens")).otherwise(col("tokens"))
)

# === Show Preprocessed Data ===
print("=== Preprocessed Data ===")
df.select("_id", "language", "filtered_tokens", "timestamp").show(truncate=False)

# === Step 2: CountVectorizer + IDF ===

# Use CountVectorizer instead of HashingTF
cv = CountVectorizer(inputCol="filtered_tokens", outputCol="rawFeatures", vocabSize=1000, minDF=1)
cv_model = cv.fit(df)
featurizedData = cv_model.transform(df)

# IDF to reweight
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# === Show featurized data ===
print("=== CountVectorized + IDF Features ===")
rescaledData.select("_id", "language", "features").show(truncate=False)

# === Step 3: LDA Topic Modeling ===

# Only keep documents that have features
filtered = rescaledData.filter(rescaledData.features.isNotNull())

# Train LDA model
lda = LDA(k=5, maxIter=10, featuresCol="features", seed=42)
lda_model = lda.fit(filtered)

# Describe topics
topics = lda_model.describeTopics(5)

print("=== Raw LDA Topics (word indices) ===")
topics.show(truncate=False)

# === Map termIndices to real words ===
vocab = cv_model.vocabulary  # Vocabulary words

# Define UDF to map indices to words
@udf(ArrayType(StringType()))
def decode_terms(termIndices):
    return [vocab[idx] for idx in termIndices]

# Add real word terms
topics_with_words = topics.withColumn("terms", decode_terms(col("termIndices")))

print("=== LDA Topics (Real Words) ===")
topics_with_words.select("topic", "terms").show(truncate=False)
