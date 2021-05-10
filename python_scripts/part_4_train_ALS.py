"""
/spark2.4/bin/pyspark
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.types import IntegerType, FloatType, DateType, StructType, StructField, StringType
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import time

sc = SparkContext
# sc.setCheckpointDir('checkpoint')
spark = SparkSession.builder.appName("gogin_spark").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

user_path = "hdfs://bigdataanalytics2-head-shdpt-v31-1-0.novalocal:8020/user/305_koryagin/"

# Create DF
data = spark.read.parquet(user_path + "input_csv_for_recommend_system/data.parquet")
data = data \
    .select('contact_id', 'product_id', 'quantity') \
    .withColumn('quantity', F.when(F.col("quantity") != 1, 1).otherwise(F.col("quantity"))) \
    .withColumnRenamed(existing='product_id', new='item_id') \
    .withColumnRenamed(existing='contact_id', new='user_id')

# Sample 20% of data
data = data.sample(fraction=0.2, seed=5)

# Basic statistics of data
numerator = data.select("quantity").count()
num_users = data.select("user_id").distinct().count()
num_items = data.select("item_id").distinct().count()
denominator = num_users * num_items
sparsity = (1.0 - (numerator * 1.0) / denominator) * 100

statistics = spark.createDataFrame(data=[('total number of rows', str('{0:,}'.format(numerator).replace(',', '\''))),
                                         ('number of users', str('{0:,}'.format(num_users).replace(',', '\''))),
                                         ('number of items', str('{0:,}'.format(num_items).replace(',', '\''))),
                                         ('sparsity', str(sparsity)[:5] + "% empty")],
                                   schema=['statistic', 'value'])
statistics.show(truncate=False)
"""
+--------------------+------------+
|statistic           |value       |
+--------------------+------------+
|total number of rows|3'869'063   |
|number of users     |1'130'779   |
|number of items     |21'960      |
|sparsity            |99.98% empty|
+--------------------+------------+
"""
# Create test and train set
(train, test) = data.randomSplit(weights=[0.9, 0.1], seed=3)

# Create ALS model
als = ALS(userCol="user_id", itemCol="item_id", ratingCol="quantity",
          nonnegative=True, implicitPrefs=True, coldStartStrategy="drop")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="quantity", predictionCol="prediction")

start = time.time()
model = als.fit(train)
end = time.time()

print('time = ' + str(end - start))

# Save model
# TODO: Не работает сохранение модели
# model.write().overwrite().save(user_path + "ml_models/my_als_model_2021-05-05_samlpe_20_percents")

# Complete the code below to extract the ALS model parameters
spark.createDataFrame(
    data=[('Rank', str(model.rank)), ('MaxIter', str(als.getMaxIter())), ('RegParam', str(als.getRegParam()))],
    schema=['parameter', 'value']).show()

# View the predictions
test_predictions = model.transform(test)
RMSE = evaluator.evaluate(test_predictions)
print('RMSE = ' + str(round(RMSE, 4)))  # RMSE = 0.9862

# Создадим таблицу с реальными и предсказанными товарами
train_actual_items = train \
    .select('user_id', 'item_id') \
    .groupBy('user_id').agg(F.collect_list(col='item_id')) \
    .withColumnRenamed(existing='collect_list(item_id)', new='actual')

train_recs_items = model.recommendForAllUsers(numItems=5) \
    .select('user_id', F.col("recommendations.item_id").alias('recs_ALS'))

result = train_actual_items.join(other=train_recs_items, on='user_id', how='inner')
result.printSchema()
result.show(n=10, truncate=True)

# Метрики качества
rdd = result.select('actual', 'recs_ALS').rdd.map(tuple)
metrics = RankingMetrics(rdd)

metrics = spark.createDataFrame(data=[('precision@k', metrics.precisionAt(5)), ('ndcg@k', metrics.ndcgAt(5)),
                                      ('meanAVGPrecision', metrics.meanAveragePrecision)], schema=['metric', 'value'])
metrics.withColumn('value', F.round('value', 5)).show(truncate=False)
"""
+----------------+-------+
|metric          |value  |
+----------------+-------+
|precision@k     |0.04911|
|ndcg@k          |0.05935|
|meanAVGPrecision|0.03411|
+----------------+-------+
"""

# TODO: Настроить перевзвешивание tf или брать sum(quantity) / max(sum(quantity) over users)

train_recs_final = model.recommendForAllUsers(numItems=5) \
    .withColumn(colName="rec_exp", col=F.explode("recommendations")) \
    .select('user_id', F.col("rec_exp.item_id"), F.col("rec_exp.rating"))
train_recs_final.show(n=10, truncate=False)
"""
+-------+-------+------------+
|user_id|item_id|rating      |
+-------+-------+------------+
|471    |41096  |0.09270032  |
|471    |140683 |0.061058853 |
|471    |136478 |0.04728449  |
|471    |104501 |0.044267062 |
|471    |140162 |0.039497897 |
|496    |41096  |0.004053822 |
|496    |140162 |0.0028327815|
|496    |46797  |0.0028049482|
|496    |140683 |0.002665252 |
|496    |32834  |0.002659647 |
+-------+-------+------------+
"""