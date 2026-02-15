from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when
from pyspark.mllib.evaluation import MulticlassMetrics
import pymongo
import pandas as pd

# Initialize Spark session
spark = SparkSession.builder \
    .appName("TopicDriftPrediction") \
    .config("spark.mongodb.input.uri", "mongodb://127.0.0.1:27017/your_database.your_collection") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1:27017/your_database.your_collection") \
    .getOrCreate()

# Step 1: Load data from MongoDB
# Replace 'your_database' and 'your_collection' with your actual MongoDB database and collection names
client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
db = client["your_database"]
collection = db["your_collection"]

# Fetch topic distributions and drift scores
# Assuming your MongoDB collection has documents with fields: timestamp, topic_distributions, drift_scores
mongo_data = list(collection.find())
df = pd.DataFrame(mongo_data)

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Step 2: Feature Engineering
# Assuming topic_distributions is a list of probabilities [topic_0, topic_1, topic_2]
# and drift_scores is a list of drift scores for each topic [drift_0, drift_1, drift_2]
# Extract topic distributions and drift scores into separate columns
spark_df = spark_df.withColumn("topic_0", col("topic_distributions")[0].cast("float")) \
                   .withColumn("topic_1", col("topic_distributions")[1].cast("float")) \
                   .withColumn("topic_2", col("topic_distributions")[2].cast("float")) \
                   .withColumn("drift_0", col("drift_scores")[0].cast("float")) \
                   .withColumn("drift_1", col("drift_scores")[1].cast("float")) \
                   .withColumn("drift_2", col("drift_scores")[2].cast("float"))

# Create a label: Drift = 1 if any drift score > 0.1, else 0
spark_df = spark_df.withColumn("drift_label", 
                               when((col("drift_0") > 0.1) | (col("drift_1") > 0.1) | (col("drift_2") > 0.1), 1).otherwise(0))

# Step 3: Prepare features for the model
feature_cols = ["topic_0", "topic_1", "topic_2", "drift_0", "drift_1", "drift_2"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(spark_df)

# Step 4: Split into train and test sets
train, test = data.randomSplit([0.8, 0.2], seed=42)

# Step 5: Train a Logistic Regression model
lr = LogisticRegression(labelCol="drift_label", featuresCol="features")
model = lr.fit(train)

# Step 6: Make predictions on the test set
predictions = model.transform(test)

# Step 7: Evaluate the model
# ROC-AUC
evaluator_roc = BinaryClassificationEvaluator(labelCol="drift_label", metricName="areaUnderROC")
auc = evaluator_roc.evaluate(predictions)
print(f"ROC-AUC: {auc}")

# Classification Report (Precision, Recall, F1-Score)
evaluator_multi = MulticlassClassificationEvaluator(labelCol="drift_label", predictionCol="prediction")
accuracy = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "accuracy"})
precision = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedPrecision"})
recall = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "weightedRecall"})
f1 = evaluator_multi.evaluate(predictions, {evaluator_multi.metricName: "f1"})

print("\nClassification Report:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

# Step 8: Compute the Confusion Matrix
# Convert predictions to RDD for MulticlassMetrics
prediction_and_label = predictions.select("prediction", "drift_label").rdd.map(lambda row: (float(row[0]), float(row[1])))
metrics = MulticlassMetrics(prediction_and_label)

# Get the confusion matrix
confusion_matrix = metrics.confusionMatrix().toArray()
print("\nConfusion Matrix:")
print(confusion_matrix)

# Step 9: Stop the Spark session
spark.stop()