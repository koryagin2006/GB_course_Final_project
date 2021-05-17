"""
/spark2.4/bin/pyspark
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
import time

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"

# Create DF
data = spark.read.parquet(user_path + "input_csv_for_recommend_system/data.parquet")
data = data \
    .select('sale_date_date', 'contact_id', 'product_id', 'quantity') \
    .withColumn('quantity', F.when(F.col("quantity") != 1, 1).otherwise(F.col("quantity"))) \
    .withColumnRenamed(existing='product_id', new='item_id') \
    .withColumnRenamed(existing='contact_id', new='user_id') \
    .withColumn('week_of_year', F.weekofyear(F.col('sale_date_date')))


def sample_by_week(df, week_col_name, split_size_weeks):
    threshold_week = int(data.select(F.max(week_col_name)).collect()[0][0]) - split_size_weeks
    df_before = df.filter(F.col(week_col_name) < threshold_week)
    df_after = df.filter(F.col(week_col_name) >= threshold_week)
    return df_before, df_after


# Отберем только 15 последних недель для обучения
before, after = sample_by_week(df=data, week_col_name='week_of_year', split_size_weeks=15)
data = after

# Create test and train set
(train, test) = data.randomSplit(weights=[0.9, 0.1], seed=3)

# Create ALS model
als = ALS(userCol="user_id", itemCol="item_id", ratingCol="quantity",
          nonnegative=True, implicitPrefs=True, coldStartStrategy="drop")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="quantity", predictionCol="prediction")

start = time.time()
model = als.fit(train)
print('time = ' + str(time.time() - start))  # time = 52.652

# Save model -- Не работает сохранение модели
# model.write().overwrite().save(path=user_path + "ml_models/my_als_model_05_13")

"""
ERROR util.Utils: Aborting task org.apache.hadoop.hdfs.protocol.DSQuotaExceededException: The DiskSpace quota of 
/user/305_koryagin is exceeded: quota = 1073741824 B = 1 GB but diskspace consumed = 1112998925 B = 1.04 GB
"""

# Create a table with real and predicted products
train_actual_items = train \
    .select('user_id', 'item_id') \
    .groupBy('user_id').agg(F.collect_list(col='item_id')) \
    .withColumnRenamed(existing='collect_list(item_id)', new='actual')

train_recs_items = model.recommendForAllUsers(numItems=5) \
    .select('user_id', F.col("recommendations.item_id").alias('recs_ALS'))

result = train_actual_items.join(other=train_recs_items, on='user_id', how='inner')

# Quality metrics
test_predictions = model.transform(test)
metrics = RankingMetrics(predictionAndLabels=result.select('actual', 'recs_ALS').rdd.map(tuple))
metrics_df = spark.createDataFrame(data=[('RMSE', evaluator.evaluate(test_predictions)),
                                         ('precision@k', metrics.precisionAt(5)),
                                         ('ndcg@k', metrics.ndcgAt(5)),
                                         ('meanAVGPrecision', metrics.meanAveragePrecision)],
                                   schema=['metric', 'value'])

metrics_df.withColumn('value', F.round('value', 5)).show(truncate=False)
