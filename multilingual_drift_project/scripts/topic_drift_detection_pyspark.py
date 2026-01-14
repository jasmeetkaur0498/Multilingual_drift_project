from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split, udf, to_date, when
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.clustering import LDA

# === Create Spark Session ===
spark = SparkSession.builder \
    .appName("TopicDriftDetection") \
    .config("spark.mongodb.read.connection.uri", "mongodb://127.0.0.1/topic_drift.documents") \
    .config("spark.jars", 
            "/Users/jasmeetkaur/Downloads/mongo-spark-connector_2.12-10.2.1.jar,"
            "/Users/jasmeetkaur/Downloads/mongodb-driver-sync-4.11.1.jar,"
            "/Users/jasmeetkaur/Downloads/mongodb-driver-core-4.11.1.jar,"
            "/Users/jasmeetkaur/Downloads/bson-4.11.1.jar") \
    .getOrCreate()

# === Load Data ===
df = spark.read.format("mongodb").load()

# Preprocessing
df = df.withColumn("text_clean", lower(col("text")))
df = df.withColumn(
    "text_clean",
    when(col("language") == "en", regexp_replace(col("text_clean"), r"[^a-zA-Z\s]", ""))
    .otherwise(col("text_clean"))
)
df = df.withColumn("tokens", split(col("text_clean"), " "))

@udf(ArrayType(StringType()))
def remove_empty_tokens(words):
    return [word for word in words if word.strip() != ""]

df = df.withColumn("tokens", remove_empty_tokens(col("tokens")))

remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
df = remover.transform(df)

df = df.withColumn(
    "filtered_tokens",
    when(col("language") == "en", col("filtered_tokens")).otherwise(col("tokens"))
)

# === Add a Date Column ===
df = df.withColumn("date", to_date(col("timestamp")))

# === Group by Date and Detect Topics ===
dates = df.select("date").distinct().collect()

for row in dates:
    date = row["date"]
    print(f"\n=== Topics on {date} ===")

    daily_df = df.filter(col("date") == date)

    # CountVectorizer
    cv = CountVectorizer(inputCol="filtered_tokens", outputCol="rawFeatures", vocabSize=1000, minDF=1)
    cv_model = cv.fit(daily_df)
    featurizedData = cv_model.transform(daily_df)

    # IDF
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    filtered = rescaledData.filter(rescaledData.features.isNotNull())

    if filtered.count() < 2:
        print("Not enough data for LDA on this date.")
        continue

    # LDA
    lda = LDA(k=3, maxIter=10, featuresCol="features", seed=42)
    lda_model = lda.fit(filtered)

    topics = lda_model.describeTopics(5)
    vocab = cv_model.vocabulary

    @udf(ArrayType(StringType()))
    def decode_terms(termIndices):
        return [vocab[idx] for idx in termIndices]

    topics = topics.withColumn("terms", decode_terms(col("termIndices")))

    topics.select("topic", "terms").show(truncate=False)
